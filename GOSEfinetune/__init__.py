from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, RobertaConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .models.LiLTRobertaLike import (
    LiLTRobertaLikeConfig,
    LiLTRobertaLikeForRelationExtraction,
    LiLTRobertaLikeForTokenClassification,
    LiLTRobertaLikeTokenizer,
    LiLTRobertaLikeTokenizerFast,
)
from .models.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForRelationExtraction,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
)
from .models.layoutxlm import (
    LayoutXLMConfig,
    LayoutXLMForRelationExtraction,
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
    LayoutXLMTokenizerFast,
)

CONFIG_MAPPING.update([("liltrobertalike", LiLTRobertaLikeConfig),("layoutlmv2", LayoutLMv2Config), ("layoutxlm", LayoutXLMConfig)])
MODEL_NAMES_MAPPING.update([("liltrobertalike", "LiLTRobertaLike"),("layoutlmv2", "LayoutLMv2"), ("layoutxlm", "LayoutXLM")])
TOKENIZER_MAPPING.update(
    [
        (LiLTRobertaLikeConfig, (LiLTRobertaLikeTokenizer, LiLTRobertaLikeTokenizerFast)),
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast)),
    ]
)

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'
if TAG == 'monolingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": RobertaConverter,"LayoutLMv2Tokenizer": BertConverter, "LayoutXLMTokenizer": XLMRobertaConverter})
elif TAG == 'multilingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": XLMRobertaConverter,"LayoutLMv2Tokenizer": BertConverter, "LayoutXLMTokenizer": XLMRobertaConverter})

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForTokenClassification),(LayoutLMv2Config, LayoutLMv2ForTokenClassification), (LayoutXLMConfig, LayoutXLMForTokenClassification)]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForRelationExtraction),(LayoutLMv2Config, LayoutLMv2ForRelationExtraction), (LayoutXLMConfig, LayoutXLMForRelationExtraction)]
)

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForRelationExtraction = auto_class_factory(
    "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
)
