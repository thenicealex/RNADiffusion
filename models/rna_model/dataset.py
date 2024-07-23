from typing import List, Any, Optional, Collection, Tuple, Dict
from pathlib import Path
import torch

from models.rna_model.evo.ffindex import MSAFFindex
from models.rna_model.evo.tokenization import Vocab
from models.rna_model.evo.typed import PathLike
from models.rna_model.evo.dataset import CollatableVocabDataset, NPZDataset, JsonDataset, A3MDataset, FastaDataset
from models.rna_model.evo.tensor import collate_tensors
from models.rna_model.rna_esm.data import Alphabet


class RNADataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)

        self.rna_id = split_files
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method,
        )

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rna_id = self.rna_id[index]
        msa = self.a3m_data[index]
        tokens = torch.from_numpy(self.vocab.encode(msa))

        return rna_id, tokens


class LMDataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)
        self.rnaids = split_files
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method  # "fast", "best"
        )
        self.npz_data = NPZDataset(
            data_path / "npz", split_files=split_files, lazy=True
        )
        assert len(self.a3m_data) == len(self.npz_data)

    def get(self, key: str):
        msa = self.a3m_data.get(key)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        missing_nt_index = torch.from_numpy(self.npz_data[key]['missing_nt_index'])
        contacts = torch.from_numpy(self.npz_data[key]['olabel'])
        return tokens, contacts, missing_nt_index

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rnaid = self.rnaids[index]
        msa = self.a3m_data[index]
        # msa = FastaDataset.rna_trans_protein(msa)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        contacts = torch.from_numpy(self.npz_data[index]['olabel'])
        missing_nt_index = torch.from_numpy(self.npz_data[index]['missing_nt_index']).type(torch.long)
        return rnaid, tokens, contacts, missing_nt_index

    def collater(
            self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        rnaid, tokens, contacts, missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'rna_id': rnaid,
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result


class TSDataset(CollatableVocabDataset):
    def __init__(
        self,
        data_path: PathLike,
        msa_path: PathLike,
        vocab: Vocab,
        split_files: Optional[Collection[str]] = None,
        json_file: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = 1,
        sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method  #"hhfilter", "sample-weights", "diversity-max", "diversity-min"
        )
        self.json_data = JsonDataset(
            data_path=data_path,
            split_files=split_files,
            json_file=json_file,
        )
        self.rna_id = split_files
        assert len(self.a3m_data) == len(self.json_data)


    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rna_id = self.rna_id[index]
        msa = self.a3m_data[index]
        tokens = torch.from_numpy(self.vocab.encode(msa))
        contacts = torch.tensor(self.json_data[index][3])
        missing_nt_index = torch.tensor(self.json_data[index][4]).type(torch.long)
        return rna_id, tokens, contacts, missing_nt_index

    def collater(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        tokens,contacts,missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result


class MSADataset(CollatableVocabDataset):
    def __init__(self, ffindex_path: PathLike):
        vocab = Vocab.from_esm_alphabet(
            Alphabet.from_architecture("MSA Transformer")
        )
        super().__init__(vocab)

        ffindex_path = Path(ffindex_path)
        index_file = ffindex_path.with_suffix(".ffindex")
        data_file = ffindex_path.with_suffix(".ffdata")
        self.ffindex = MSAFFindex(index_file, data_file)

    def __len__(self):
        return len(self.ffindex)

    def __getitem__(self, idx):
        msa = self.ffindex[idx]
        return torch.from_numpy(self.vocab.encode(msa))

    def collater(self, batch: List[Any]) -> Any:
        return collate_tensors(batch)


class TRRosettaContactDataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        self.a3m_data = A3MDataset(
            data_path / "a3m",
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
        )
        self.npz_data = NPZDataset(
            data_path / "npz", split_files=split_files, lazy=True
        )

        assert len(self.a3m_data) == len(self.npz_data)

    def get(self, key: str):
        msa = self.a3m_data.get(key)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        distogram = self.npz_data.get(key)["dist6d"]
        contacts = (distogram > 0) & (distogram < 8)
        contacts = torch.from_numpy(contacts)
        return tokens, contacts

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        msa = self.a3m_data[index]
        tokens = torch.from_numpy(self.vocab.encode(msa))
        distogram = self.npz_data[index]["dist6d"]
        contacts = (distogram > 0) & (distogram < 8)
        contacts = torch.from_numpy(contacts)
        return tokens, contacts

    def collater(
            self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        tokens, contacts = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            "src_tokens": src_tokens,
            "tgt": targets,
            "tgt_lengths": src_lengths,
        }

        return result
