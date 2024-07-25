# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.diffusion_multinomial import MultinomialDiffusion
from models.layers import SegmentationUnet2DCondition
from models.rna_model.config import TransformerConfig, OptimizerConfig
from models.rna_model.model_slim import ESM2
from models.rna_model import rna_esm
from models.rna_model.evo.tokenization import Vocab, mapdict

CH_FOLD = 1


def get_model_id(args):
    return "multinomial_diffusion"


class DiffusionRNA2dPrediction(nn.Module):

    def __init__(
        self, num_classes, diffusion_dim, cond_dim, diffusion_steps, dp_rate, esm_ckpt,device
    ):
        super(DiffusionRNA2dPrediction, self).__init__()

        self.num_classes = num_classes
        self.diffusion_dim = diffusion_dim
        self.cond_dim = cond_dim
        self.diffusion_steps = diffusion_steps
        self.dp_rate = dp_rate
        self.esm_ckpt = esm_ckpt
        self.device = device

        # condition
        self.esm_conditioner = RNAESM2(esm_ckpt=self.esm_ckpt, device=self.device)
        self.rna_alphabet = self.esm_conditioner.rna_alphabet

        self.denoise = SegmentationUnet2DCondition(
            num_classes=self.num_classes,
            dim=self.diffusion_dim,
            cond_dim=self.cond_dim,
            num_steps=self.diffusion_steps,
            dim_mults=(1, 2, 4, 8),
            dropout=self.dp_rate,
        )

        self.diffusion = MultinomialDiffusion(
            self.num_classes, self.diffusion_steps, self.denoise
        )

    def get_alphabet(self):
        return self.rna_alphabet

    def forward(
        self,
        x_0,
        data_seq_raw,
        set_max_len,
        data_seq_encoding,
    ):
        esm_condition = self.esm_conditioner(data_seq_raw, set_max_len)

        loss = self.diffusion(x_0, esm_condition, data_seq_encoding)

        loglik_bpd = -loss.sum() / (math.log(2) * x_0.shape.numel())
        return loglik_bpd

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        data_seq_raw,
        set_max_len,
        contact_masks,
        seq_encoding,
    ):
        esm_condition = self.esm_conditioner(data_seq_raw, set_max_len)

        pred_x_0, model_prob = self.diffusion.sample(
            num_samples,
            esm_condition,
            contact_masks,
            set_max_len,
            seq_encoding,
        )

        return pred_x_0, model_prob

    @torch.no_grad()
    def sample_chain(
        self,
        num_samples,
        data_seq_raw,
        set_max_len,
        contact_masks,
        seq_encoding,
    ):
        esm_condition = self.esm_conditioner(data_seq_raw, set_max_len)

        pred_x_0_chain, model_prob_chain, pred_x_0, model_prob = (
            self.diffusion.sample_chain(
                num_samples,
                esm_condition,
                contact_masks,
                set_max_len,
                seq_encoding,
            )
        )
        return pred_x_0_chain, model_prob_chain, pred_x_0, model_prob


class RNAESM2(nn.Module):
    def __init__(self, esm_ckpt, device="cpu"):
        super(RNAESM2, self).__init__()
        self.device = device
        if esm_ckpt is None:
            raise ValueError("Please provide a valid RNA-ESM2 checkpoint")
        self.esm_ckpt = esm_ckpt
        self.model, self.rna_map_vocab, self.rna_alphabet = self.__init_model__()

    def __init_model__(self):
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
        print(f"Loading RNA-ESM2 model: {self.esm_ckpt}")
        model.load_state_dict(
            torch.load(self.esm_ckpt, map_location="cpu")["state_dict"],
            strict=True,
        )
        return model, rna_map_vocab, rna_alphabet

    def forward(self, data_seq_raw, set_max_len=80):
        self.model.eval()
        output = dict()
        self.model.to(self.device)

        with torch.no_grad():
            tokens = torch.from_numpy(self.rna_map_vocab.encode(data_seq_raw))
            infer = self.model(
                tokens.to(self.device), repr_layers=[30], return_contacts=True
            )
            embedding = infer["representations"][30]

            attention = infer["attentions"]
            b, l, n, l1, l2 = attention.shape
            attention = attention.reshape(b, l * n, l1, l2)[:, :, 1:-1, 1:-1]
            padding_value = 0
            padding_size = (
                0,
                set_max_len - attention.shape[-2],
                0,
                set_max_len - attention.shape[-1],
            )
            attention = F.pad(attention, padding_size, "constant", value=padding_value)

            start_idx = int(self.rna_map_vocab.prepend_bos)
            end_idx = embedding.size(-2) - int(self.rna_map_vocab.append_eos)
            embedding = embedding[:, start_idx:end_idx, :]
            embedding_pad = torch.zeros(
                embedding.shape[0], set_max_len - embedding.shape[1], embedding.shape[2]
            ).to(self.device)
            embedding = torch.cat([embedding, embedding_pad], dim=1)

            try:
                embedding = F.softmax(embedding, dim=-1)
            except:
                ValueError("Error in softmax")

            output["embedding"] = embedding
            output["attention"] = attention

        return output
