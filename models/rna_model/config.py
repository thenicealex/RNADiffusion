from os.path import join
from typing import Tuple
from dataclasses import dataclass


model_path = "/home/fkli/RNAm"
model_name = "RNA-ESM2-trans-2a100-mappro-KDNY-epoch=06-valid_F1=0.564-v1.ckpt"

@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 4e-4
    weight_decay: float = 1e-2  # 3e-4
    lr_scheduler: str = "warmup_linear"  # "warmup_cosine"
    warmup_steps: int = 16000  # 16000
    adam_betas: Tuple[float, float] = (0.9, 0.98)  # 0.9, 0.999
    max_steps: int = 10000000


@dataclass
class TransformerConfig:
    embed_dim: int = 640
    num_attention_heads: int = 20
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_type: str = "standard"
    performer_attention_features: int = 256
    num_layers: int = 30
    max_seqlen: int = 1024

@dataclass
class DataConfig:
    architecture: str = "rna-esm"
    num_workers: int = 16
    model_path: str = join(model_path, model_name)
    # device: str = "cpu"
    device: str = "cuda:0"


@dataclass
class TrainConfig:
    pass


@dataclass
class LoggingConfig:
    pass


@dataclass
class ProduceConfig:
    pass


@dataclass
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: TransformerConfig = TransformerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    logging: LoggingConfig = LoggingConfig()
    produce: ProduceConfig = ProduceConfig()