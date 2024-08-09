# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion_multinomial import MultinomialDiffusion
from models.layers import SegmentationUnet2DCondition
from models.rna_model.config import TransformerConfig, OptimizerConfig
from models.unet.unet_model import UNet
from models.rna_model.model_slim import ESM2
from models.rna_model import rna_esm
from models.rna_model.evo.tokenization import Vocab, mapdict


CH_FOLD = 1


class DiffusionRNA2dPrediction(nn.Module):
    def __init__(
        self,
        num_classes,
        diffusion_dim,
        cond_dim,
        diffusion_steps,
        dropout_rate,
        unet_checkpoint,
        esm_checkpoint,
        device,
    ):
        super(DiffusionRNA2dPrediction, self).__init__()

        self.num_classes = num_classes
        self.diffusion_dim = diffusion_dim
        self.cond_dim = cond_dim
        self.diffusion_steps = diffusion_steps
        self.dropout_rate = dropout_rate
        self.unet_checkpoint = unet_checkpoint
        self.esm_checkpoint = esm_checkpoint
        self.device = device

        # Initialize conditioners
        self.esm_conditioner = RNAESM2(
            esm_checkpoint=self.esm_checkpoint, device=self.device
        )
        self.unet_conditioner = RNAUNet(
            num_channels=17,
            num_classes=1,
            cond_dim=self.cond_dim,
            device=self.device,
            unet_checkpoint=self.unet_checkpoint,
        )

        # Initialize denoise layer
        self.denoise_layer = SegmentationUnet2DCondition(
            num_classes=self.num_classes,
            dim=self.diffusion_dim,
            cond_dim=self.cond_dim,
            num_steps=self.diffusion_steps,
            dim_mults=(1, 2, 4, 8),
            dropout=self.dropout_rate,
        )

        # Initialize diffusion process
        self.diffusion = MultinomialDiffusion(
            self.num_classes, self.diffusion_steps, self.denoise_layer
        )

    def forward(
        self, x_0, base_info, data_seq_raw, contact_masks, max_len, data_seq_encoding
    ):
        esm_condition = self.esm_conditioner(data_seq_raw, max_len)
        unet_condition = self.unet_conditioner(base_info)

        loss = self.diffusion(
            x_0, esm_condition, unet_condition, contact_masks, data_seq_encoding
        )
        log_likelihood_bpd = -loss.sum() / (math.log(2) * x_0.numel())
        return log_likelihood_bpd

    @torch.no_grad()
    def sample(
        self, num_samples, base_info, data_seq_raw, max_len, contact_masks, seq_encoding
    ):
        esm_condition = self.esm_conditioner(data_seq_raw, max_len)
        unet_condition = self.unet_conditioner(base_info)

        pred_x_0, model_prob = self.diffusion.sample(
            num_samples,
            esm_condition,
            unet_condition,
            contact_masks,
            max_len,
            seq_encoding,
        )
        return pred_x_0, model_prob

    @torch.no_grad()
    def sample_chain(
        self, num_samples, base_info, data_seq_raw, max_len, contact_masks, seq_encoding
    ):
        esm_condition = self.esm_conditioner(data_seq_raw, max_len)
        unet_condition = self.unet_conditioner(base_info)

        pred_x_0_chain, model_prob_chain, pred_x_0, model_prob = (
            self.diffusion.sample_chain(
                num_samples,
                esm_condition,
                unet_condition,
                contact_masks,
                max_len,
                seq_encoding,
            )
        )
        return pred_x_0_chain, model_prob_chain, pred_x_0, model_prob


class RNAESM2(nn.Module):
    def __init__(self, esm_checkpoint, device="cuda:0"):
        super(RNAESM2, self).__init__()
        self.device = device
        if not esm_checkpoint:
            raise ValueError("Please provide a valid RNA-ESM2 checkpoint")
        self.esm_checkpoint = esm_checkpoint
        self.model, self.rna_map_vocab, self.rna_alphabet = self._initialize_model()

    def _initialize_model(self):
        _, protein_alphabet = rna_esm.pretrained.esm2_t30_150M_UR50D()
        rna_alphabet = rna_esm.data.Alphabet.from_architecture("rna-esm")
        protein_vocab = Vocab.from_esm_alphabet(protein_alphabet)
        rna_vocab = Vocab.from_esm_alphabet(rna_alphabet)
        rna_map_dict = mapdict(protein_vocab, rna_vocab)
        rna_map_vocab = Vocab.from_esm_alphabet(rna_alphabet, rna_map_dict)
        model = ESM2(
            vocab=protein_vocab,
            model_config=TransformerConfig(),
            optimizer_config=OptimizerConfig(),
            contact_train_data=None,
            token_dropout=True,
        )
        print(f"Loading RNA-ESM2 model: {self.esm_checkpoint}")
        model.load_state_dict(
            torch.load(self.esm_checkpoint, map_location="cpu")["state_dict"],
            strict=True,
        )
        return model, rna_map_vocab, rna_alphabet

    def forward(self, data_seq_raw, max_len=80):
        self.model.eval()
        self.model.to(self.device)

        output = {}
        with torch.no_grad():
            tokens = torch.from_numpy(self.rna_map_vocab.encode(data_seq_raw))
            infer = self.model(
                tokens.to(self.device), repr_layers=[30], return_contacts=True
            )
            embedding = infer["representations"][30]

            attention = infer["attentions"]
            b, l, n, l1, l2 = attention.shape
            attention = attention.reshape(b, l * n, l1, l2)[:, :, 1:-1, 1:-1]
            padding_size = (
                0,
                max_len - attention.shape[-2],
                0,
                max_len - attention.shape[-1],
            )
            attention = F.pad(attention, padding_size, "constant", value=0)

            start_idx = int(self.rna_map_vocab.prepend_bos)
            end_idx = embedding.size(-2) - int(self.rna_map_vocab.append_eos)
            embedding = embedding[:, start_idx:end_idx, :]
            embedding_pad = torch.zeros(
                embedding.shape[0], max_len - embedding.shape[1], embedding.shape[2]
            ).to(self.device)
            embedding = torch.cat([embedding, embedding_pad], dim=1)

            try:
                embedding = F.softmax(embedding, dim=-1)
            except:
                raise ValueError("Error in softmax")

            output["embedding"] = embedding
            output["attention"] = attention

        return output


class RNAUNet(nn.Module):
    def __init__(self, num_channels, num_classes, cond_dim, device, unet_checkpoint=None):
        super(RNAUNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.cond_dim = cond_dim
        self.unet_checkpoint = unet_checkpoint
        self.conv_layer = nn.Conv2d(
            int(32 * CH_FOLD), self.cond_dim, kernel_size=1, stride=1, padding=0
        )
        self.model = UNet(self.num_channels, self.num_classes)
        self.device = device
        self._initialize_model()

    def _initialize_model(self):
        self.model.load_state_dict(torch.load(self.unet_checkpoint, map_location="cpu"))
        self.model.Conv_1x1 = self.conv_layer
        self.model.requires_grad_(True)

    def forward(self, x):
        self.model.to(self.device)
        return self.model(x)
