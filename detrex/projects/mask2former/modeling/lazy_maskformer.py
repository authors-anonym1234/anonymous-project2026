"""
LazyConfig-friendly Mask2Former components that avoid Detectron2 CfgNode construction.
We reuse the core logic but make constructors explicit.
"""
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import ShapeSpec

from .meta_arch.mask_former_head import MaskFormerHead
from .transformer_decoder.maskformer_transformer_decoder import TransformerDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .matcher import HungarianMatcher
from .criterion import SetCriterion


class LazyMaskFormer(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        input_shape: Dict[str, ShapeSpec],
        num_classes: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_queries: int = 100,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1,
        deep_supervision: bool = True,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        object_mask_threshold: float = 0.8,
        overlap_threshold: float = 0.8,
        size_divisibility: int = 32,
        sem_seg_postprocess_before_inference: bool = True,
        semantic_on: bool = False,
        instance_on: bool = True,
        panoptic_on: bool = False,
        test_topk_per_image: int = 100,
        transformer_in_feature: str = "multi_scale_pixel_decoder",
        transformer_dec_layers: int = 10,
        transformer_nheads: int = 8,
        transformer_dropout: float = 0.0,
        transformer_dim_feedforward: int = 2048,
        transformer_pre_norm: bool = False,
        transformer_enc_layers: int = 0,
        pixel_decoder_common_stride: int = 4,
        pixel_decoder_transformer_in_features: List[str] = None,
        pixel_decoder_transformer_enc_layers: int = 6,
        pixel_decoder_deform_num_heads: int = 8,
        pixel_decoder_deform_num_points: int = 4,
        pixel_decoder_total_num_feature_levels: int = 4,
    ):
        super().__init__()
        pixel_decoder_transformer_in_features = (
            pixel_decoder_transformer_in_features
            if pixel_decoder_transformer_in_features is not None
            else ["res3", "res4", "res5"]
        )

        self.backbone = backbone

        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=transformer_dropout,
            transformer_nheads=transformer_nheads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=pixel_decoder_transformer_enc_layers,
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_in_features=pixel_decoder_transformer_in_features,
            common_stride=pixel_decoder_common_stride,
            num_feature_levels=len(pixel_decoder_transformer_in_features),
            total_num_feature_levels=pixel_decoder_total_num_feature_levels,
            deform_num_heads=pixel_decoder_deform_num_heads,
            deform_num_points=pixel_decoder_deform_num_points,
            feature_order="low2high",
        )

        transformer_predictor = TransformerDecoder(
            in_channels=256,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=num_queries,
            nheads=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            dec_layers=transformer_dec_layers,
            mask_dim=256,
            enforce_input_project=False,
            two_stage=False,
            dn=None,
            noise_scale=0.0,
            dn_num=0,
            initialize_box_type="mask2box",
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=pixel_decoder_total_num_feature_levels,
            dropout=transformer_dropout,
            activation="relu",
            dec_n_points=4,
            return_intermediate_dec=True,
            query_dim=4,
            dec_layer_share=False,
            semantic_ce_loss=False,
        )

        sem_seg_head = MaskFormerHead(
            input_shape=input_shape,
            num_classes=num_classes,
            pixel_decoder=pixel_decoder,
            loss_weight=1.0,
            ignore_value=255,
            transformer_predictor=transformer_predictor,
            transformer_in_feature=transformer_in_feature,
        )

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=train_num_points,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(transformer_dec_layers - 1):
                aux_weight_dict.update({f"loss_ce_{i}": class_weight,
                                        f"loss_mask_{i}": mask_weight,
                                        f"loss_dice_{i}": dice_weight})
            weight_dict.update(aux_weight_dict)

        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=["labels", "masks"],
            num_points=train_num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
        )

        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = None
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(img - self.pixel_mean) / self.pixel_std for img in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            losses = self.criterion(outputs, targets)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = F.interpolate(
                        mask_pred_result.unsqueeze(0),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                processed_results[-1]["sem_seg"] = mask_pred_result

                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result, image_size
                    )
                    processed_results[-1]["instances"] = instance_r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, (image_height, image_width) in zip(targets, images.image_sizes):
            masks = targets_per_image.get("masks", None)
            if masks is not None:
                # pad to match image size
                masks = F.pad(masks, (0, w_pad - masks.shape[-1], 0, h_pad - masks.shape[-2]))

            new_targets.append(
                {
                    "labels": targets_per_image["labels"],
                    "masks": masks,
                }
            )
        return new_targets

    def instance_inference(self, mask_cls, mask_pred, image_size):
        # adapted from original maskformer_model
        import torch.nn.functional as F
        from detectron2.modeling.postprocessing import detector_postprocess
        from detectron2.structures import Instances, Boxes

        # resize mask predictions
        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
        )[0]
        # softmax on class dimension
        mask_cls = mask_cls.sigmoid()

        scores, labels = mask_cls.max(-1)
        keep = scores > self.object_mask_threshold

        scores = scores[keep]
        labels = labels[keep]
        mask_pred = mask_pred[keep]

        instances = Instances(image_size)
        instances.pred_masks = mask_pred > 0.5
        # dummy boxes
        instances.pred_boxes = Boxes(torch.zeros(len(scores), 4, device=mask_pred.device))
        instances.scores = scores
        instances.pred_classes = labels
        return instances
