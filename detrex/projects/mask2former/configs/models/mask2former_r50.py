from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import ResNet, BasicStem

from projects.mask2former.maskformer_model import MaskFormer
from projects.mask2former.modeling.matcher import HungarianMatcher
from projects.mask2former.modeling.criterion import SetCriterion
from projects.mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from projects.mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from projects.mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# Single LazyCall with nested components
model = L(MaskFormer)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(depth=50, stride_in_1x1=False, norm="FrozenBN"),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=0,
    ),
    sem_seg_head=L(MaskFormerHead)(
        input_shape={
            "res2": L(ShapeSpec)(channels=256, stride=4),
            "res3": L(ShapeSpec)(channels=512, stride=8),
            "res4": L(ShapeSpec)(channels=1024, stride=16),
            "res5": L(ShapeSpec)(channels=2048, stride=32),
        },
        num_classes=80,
        pixel_decoder=L(MSDeformAttnPixelDecoder)(
            input_shape={
                "res2": L(ShapeSpec)(channels=256, stride=4),
                "res3": L(ShapeSpec)(channels=512, stride=8),
                "res4": L(ShapeSpec)(channels=1024, stride=16),
                "res5": L(ShapeSpec)(channels=2048, stride=32),
            },
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_in_features=["res3", "res4", "res5"],
            common_stride=4,
            use_checkpoint=False,
        ),
        loss_weight=1.0,
        ignore_value=255,
        transformer_predictor=L(MultiScaleMaskedTransformerDecoder)(
            in_channels=256,
            mask_classification=True,
            num_classes=80,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,       # 9 + the code adds the learnable query layer on top = 10 from original implementation
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            use_checkpoint=False,
        ),
        transformer_in_feature="multi_scale_pixel_decoder",
    ),
    criterion=L(SetCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            num_points=12544,
        ),
        weight_dict={
            "loss_ce": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0,
            **{f"loss_ce_{i}": 2.0 for i in range(9)},
            **{f"loss_mask_{i}": 5.0 for i in range(9)},
            **{f"loss_dice_{i}": 5.0 for i in range(9)},
        },
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    num_queries=200,
    object_mask_threshold=0.8,
    overlap_threshold=0.8,
    metadata=None,
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    semantic_on=False,
    panoptic_on=False,
    instance_on=True,
    test_topk_per_image=100,
)
