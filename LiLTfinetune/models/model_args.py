from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class GOSEArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    #==== decoder config ==========================================
    attn_lr: int = field(
        default=10, metadata={"help": "."}
    )
    rounds: int = field(
        default=4, metadata={"help": "iterative training rounds."}
    )
    decoder_name: str = field(
        default='re',metadata={"help": ""}
    )
    use_attn: bool = field(
        default=False, metadata={"help": ""}
    )
    use_prefix: bool = field(
        default=False, metadata={"help": ""}
    )
    use_gate: bool = field(
        default=False, metadata={"help": ""}
    )
    use_global_mask: bool = field(
        default=False,metadata={"help":""}
    )
    window_size : int = field(
        default=10, metadata={"help": "."}
    )
    pooling_mode: str = field(
        default='mean', metadata={"help": ""}
    )
    #==== load checkpoint and save result config =================
    use_trained_model: bool = field(
        default=False, metadata={"help": ""}
    )
    checkpoint_type: str = field(
        default='en',metadata={"help": "."}
    )
    save_result: bool = field(
        default=False,metadata = {"help": "."}
    )
    save_dir: str = field(
        default='en',metadata={"help": "."}
    )
    save_file_name: str = field(
        default='None',
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    save_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    fewshot: bool = field(
        default=False,
        metadata={"help": ""},
    )
