from detrex.config import get_config
from detectron2.config import LazyCall as L
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler

from projects.mask2former.configs.models.mask2former_swin_base import model
from detrex.projects.maskdino.configs.data.coco_instance_seg import dataloader

train = get_config("common/train.py").train
train.output_dir = "./output/mask2former_swin_base_100ep"
train.max_iter = 368750
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000
train.device = "cuda"
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.01
train.clip_grad.params.norm_type = 2

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[327778, 355092],
    ),
    warmup_length=10 / train.max_iter,
    warmup_factor=1.0,
)

optimizer = get_config("common/optim.py").AdamW
optimizer.lr = 1e-4
optimizer.weight_decay = 0.05
optimizer.betas = (0.9, 0.999)
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1 if "backbone" in module_name else 1
)

dataloader.train.total_batch_size = 16
dataloader.train.num_workers = 4

if hasattr(dataloader, "mapper"):
    dataloader.mapper.name = "coco_instance_lsj"
