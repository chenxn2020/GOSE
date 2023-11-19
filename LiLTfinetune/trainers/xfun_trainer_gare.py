import collections
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers.utils import logging
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import EvalPrediction, PredictionOutput, speed_metrics, ShardedDDPOption
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, AdamW, get_scheduler

from .funsd_trainer import FunsdTrainer

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


class XfunSerTrainer(FunsdTrainer):
    pass


class XfunReTrainer(FunsdTrainer):
    def __init__(self, attn_lr, **kwargs):
        super().__init__(**kwargs)
        self.label_names.append("relations")
        self.label_names.append("bbox")
        self.attn_lr = attn_lr

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        #from IPython import embed;embed()
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        labels = tuple(inputs.get(name) for name in self.label_names)
        return outputs, labels

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        re_labels = None
        re_bboxs = None
        pred_relations = None
        entities = None
        
        for step, inputs in enumerate(dataloader):
            outputs, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            re_labels = labels[1] if re_labels is None else re_labels + labels[1]
            pred_relations = (
                outputs.pred_relations if pred_relations is None else pred_relations + outputs.pred_relations
            )
            entities = outputs.entities if entities is None else entities + outputs.entities
            #---获取每篇文档中实体的bbox
            re_bboxs = labels[2].tolist() if re_bboxs is None else re_bboxs + labels[2].tolist()

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        gt_relations = []
        #---整理ground truth
        for b in range(len(re_labels)):
            rel_sent = []
            for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
                rel = {}
                rel["head_id"] = head
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = tail
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]
                rel["type"] = 1
                #---得到key和value之间link的坐标(左上点的link(head, tail)),用首token的bbox
                key_bbox_left_top = re_bboxs[b][entities[b]["start"][rel["head_id"]]][:2]
                value_bbox_left_top = re_bboxs[b][entities[b]["start"][rel["tail_id"]]][:2]
                rel["link"] = (tuple(key_bbox_left_top), tuple(value_bbox_left_top))

                rel_sent.append(rel)

            gt_relations.append(rel_sent)

        re_metrics = self.compute_metrics(EvalPrediction(predictions=pred_relations, label_ids=gt_relations))
        """使用的是micro F1 Scores"""
        re_metrics = {
            "precision": re_metrics["ALL"]["p"],
            "recall": re_metrics["ALL"]["r"],
            "f1": re_metrics["ALL"]["f1"],
            "cross_total_ratio": re_metrics["ALL"]["cross_total_ratio"] #link的交叉率
        }
        re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()

        metrics = {}

        # # Prefix all keys with metric_key_prefix + '_'
        for key in list(re_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = re_metrics.pop(key)
            else:
                metrics[f"{key}"] = re_metrics.pop(key)

        return metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.args.local_rank = torch.distributed.get_rank()

        start_time = time.time()

        metrics = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

    def create_optimizer(self, speedup_r=4.):
        if self.optimizer is None:
            decoder_name = None
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            for name in decay_parameters:
                if 'gru' in name:
                    decoder_name = 'gare'
                if 'rel_classifier' in name:
                    decoder_name = 'liltre'
                if decoder_name is not None:
                    break
            print(f"**********************************Decoder_Name: {decoder_name}****************************")
            if decoder_name == 'gare':
                decay_parameters = [name for name in decay_parameters if "bias" not in name and 'attn' not in name and 'gru' not in name ]
                # speedup_parameters = [name for name in get_parameter_names(self.model, []) if 'extractor' in name and 'rel_classifier' not in name]
                speedup_parameters = [name for name in get_parameter_names(self.model, []) if 'extractor' in name and ('attn' not in name and 'gru' not in name )]
                speedup_parameters_1 = [name for name in get_parameter_names(self.model, []) if ('attn' in name or 'gru' in name) and 'extractor' in name]
            
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n in speedup_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate *speedup_r,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n in speedup_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate *speedup_r,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n not in speedup_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n not in speedup_parameters and n not in speedup_parameters_1],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in speedup_parameters_1],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate*self.attn_lr,
                    },
                ]
                print(f"**********************************attn_lr: {self.args.learning_rate*self.attn_lr}****************************")
            elif  decoder_name == 'liltre':
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                speedup_parameters = [name for name in get_parameter_names(self.model, []) if 'extractor' in name and 'rel_classifier' not in name]
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n in speedup_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate *speedup_r,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n in speedup_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate *speedup_r,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n not in speedup_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n not in speedup_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ]

            # from IPython import embed;embed();exit()
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
