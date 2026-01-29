import os
import argparse

from canopyrs.engine.config_parsers.base import get_config_path

import pandas as pd
from pathlib import Path
#from experiments.resolution.evaluate.get_wandb import wandb_runs_to_dataframe

from canopyrs.engine.benchmark import SegmenterBenchmarker
from canopyrs.engine.config_parsers.aggregator import AggregatorConfig
from canopyrs.engine.config_parsers.detector import DetectorConfig
from canopyrs.engine.config_parsers.segmenter import SegmenterConfig

from canopyrs.config import config_folder_path


def evaluate(run_ids, best_run_id, models_root_path, raw_data_root, output_path, valid_nms_datasets, test_datasets, rf1_eval_iou_threshold, nms_algorithm, prompted_segmenter_config, detector_config_path, seed_for=None):
    output_path = Path(output_path)
    
    output_path_valid_run = Path(output_path) / 'valid' / best_run_id
    output_path_valid_run.mkdir(parents=True, exist_ok=True)

    valid_benchmarker = SegmenterBenchmarker(
        output_folder=output_path_valid_run,
        fold_name='valid',
        raw_data_root=raw_data_root,
        eval_iou_threshold=rf1_eval_iou_threshold
    )
    aggregator_base = AggregatorConfig(
        nms_algorithm=nms_algorithm
    )

    nms_ious = [i / 20 for i in range(1, 21)]
    nms_scores = [i / 20 for i in range(1, 21)]

    if prompted_segmenter_config:
        # the segmenter is prompted
        if detector_config_path is None:
            detector_config = DetectorConfig.from_yaml(f'{models_root_path}/{best_run_id}/config.yaml')
        else:
            detector_config = DetectorConfig.from_yaml(detector_config_path)
        #Maybe i want a checkpoint path, maybe not. see if there is one in the path
        if seed_for == 'detector':
            detector_checkpoint_path = f'{models_root_path}/{best_run_id}/model_best.pth'
            if Path(detector_checkpoint_path).exists():
                print(f'Using detector checkpoint: {detector_checkpoint_path}')
                detector_config.checkpoint_path = detector_checkpoint_path
            else:
                print(f'No detector checkpoint found at: {detector_checkpoint_path}')
        elif seed_for == 'segmenter':
            segmenter_checkpoint_path = f'{models_root_path}/{best_run_id}/model_best.pt'
            if Path(segmenter_checkpoint_path).exists():
                print(f'Using segmenter checkpoint: {segmenter_checkpoint_path}')
                prompted_segmenter_config.checkpoint_path = segmenter_checkpoint_path
            else:
                print(f'No segmenter checkpoint found at: {segmenter_checkpoint_path}')

        best_aggregator_config = valid_benchmarker.find_optimal_nms_iou_threshold(
            prompter_detector_config=detector_config,
            segmenter_config=prompted_segmenter_config,
            base_aggregator_config=aggregator_base,
            dataset_names=valid_nms_datasets,
            nms_iou_thresholds=nms_ious,
            nms_score_thresholds=nms_scores,
            n_workers=12
        )

    else:
        # it's an end-to-end segmenter
        segmenter_config = SegmenterConfig.from_yaml(f'{models_root_path}/{best_run_id}/config.yaml')
        segmenter_checkpoint_path = f'{models_root_path}/{best_run_id}/model_best.pth'
        if Path(segmenter_checkpoint_path).exists():
            segmenter_config.checkpoint_path = segmenter_checkpoint_path

        best_aggregator_config = valid_benchmarker.find_optimal_nms_iou_threshold(
            segmenter_config=segmenter_config,
            base_aggregator_config=aggregator_base,
            dataset_names=valid_nms_datasets,
            nms_iou_thresholds=nms_ious,
            nms_score_thresholds=nms_scores,
            n_workers=12
        )

    tile_level_results = []
    raster_level_results = []

    for run_id in run_ids:
        test_fold = 'test'
        output_path_run = Path(output_path) / run_id
        output_path_run.mkdir(parents=True, exist_ok=True)

        test_benchmarker = SegmenterBenchmarker(
            output_folder=output_path_run,
            fold_name=test_fold,
            raw_data_root=raw_data_root,
            eval_iou_threshold=rf1_eval_iou_threshold
        )

        aggregator_config = best_aggregator_config.model_copy(deep=True)

        if prompted_segmenter_config:
            if detector_config_path is None:
                detector_config = DetectorConfig.from_yaml(f'{models_root_path}/{run_id}/config.yaml')
            else:
                detector_config = DetectorConfig.from_yaml(detector_config_path)
            if seed_for == 'detector':
                detector_checkpoint_path = f'{models_root_path}/{run_id}/model_best.pth'
                if Path(detector_checkpoint_path).exists():
                    print(f'Using detector checkpoint: {detector_checkpoint_path}')
                    detector_config.checkpoint_path = detector_checkpoint_path
                else:
                    print(f'No detector checkpoint found at: {detector_checkpoint_path}')
            elif seed_for == 'segmenter':
                segmenter_checkpoint_path = f'{models_root_path}/{run_id}/model_best.pt'
                if Path(segmenter_checkpoint_path).exists():
                    print(f'Using segmenter checkpoint: {segmenter_checkpoint_path}')
                    prompted_segmenter_config.checkpoint_path = segmenter_checkpoint_path
                else:
                    print(f'No segmenter checkpoint found at: {segmenter_checkpoint_path}')
            segmenter_checkpoint_path = f'{models_root_path}/{best_run_id}/model_best.pth'
            tlm, rlm = test_benchmarker.benchmark(
                prompter_detector_config=detector_config,
                segmenter_config=prompted_segmenter_config,
                aggregator_config=aggregator_config,
                dataset_names=test_datasets,
            )
        else:
            segmenter_config = SegmenterConfig.from_yaml(f'{models_root_path}/{run_id}/config.yaml')
            segmenter_checkpoint_path = f'{models_root_path}/{run_id}/model_best.pth'
            if Path(segmenter_checkpoint_path).exists():
                segmenter_config.checkpoint_path = segmenter_checkpoint_path
            tlm, rlm = test_benchmarker.benchmark(
                segmenter_config=segmenter_config,
                aggregator_config=aggregator_config,
                dataset_names=test_datasets,
            )
        tlm['run_id'] = run_id
        rlm['run_id'] = run_id

        tile_level_results.append(tlm)
        raster_level_results.append(rlm)

    summary_tile = SegmenterBenchmarker.compute_mean_std_metric_tables(
        tile_level_results,
        output_path / 'tile_level_summary.csv'
    )

    summary_raster = SegmenterBenchmarker.compute_mean_std_metric_tables(
        raster_level_results,
        output_path / 'raster_level_summary.csv'
    )

    combined = SegmenterBenchmarker.merge_tile_and_raster_summaries(
        summary_tile,
        summary_raster,
        output_csv=Path(output_path) / 'combined_level_summary.csv',
        tile_prefix='tile',
        raster_prefix='raster'
    )


def evaluate_external_method(external_method_name,
                             external_method_config_name,
                             raw_data_root,
                             output_path,
                             valid_nms_datasets,
                             test_datasets,
                             rf1_eval_iou_threshold,
                             nms_algorithm='iou'):
    """
    Evaluate a single external detector (one seed).
    - config_path: path to SegmenterConfig YAML (with correct checkpoint_path inside)
    - output_path: base output dir (same role as in evaluate)
    """
    output_path = Path(output_path)

    # ---------- 1) Find best NMS params on validation ----------
    segmenter_config = SegmenterConfig.from_yaml(str(config_folder_path / 'default_components' / external_method_config_name))

    output_path_valid_run = output_path / 'valid' / external_method_name
    output_path_valid_run.mkdir(parents=True, exist_ok=True)

    valid_benchmarker = SegmenterBenchmarker(
        output_folder=output_path_valid_run,
        fold_name='valid',
        raw_data_root=raw_data_root,
        eval_iou_threshold=rf1_eval_iou_threshold,
    )

    aggregator_base = AggregatorConfig(
        nms_algorithm=nms_algorithm,
    )

    best_aggregator_config = valid_benchmarker.find_optimal_nms_iou_threshold(
        segmenter_config=segmenter_config,
        base_aggregator_config=aggregator_base,
        dataset_names=valid_nms_datasets,
        nms_iou_thresholds=[i / 20 for i in range(1, 21)],
        nms_score_thresholds=[i / 20 for i in range(1, 21)],
        n_workers=12,
    )

    # ---------- 2) Run on test datasets with best NMS ----------
    test_fold = 'test'
    output_path_run = output_path / external_method_name
    output_path_run.mkdir(parents=True, exist_ok=True)

    test_benchmarker = SegmenterBenchmarker(
        output_folder=output_path_run,
        fold_name=test_fold,
        raw_data_root=raw_data_root,
        eval_iou_threshold=rf1_eval_iou_threshold,
    )

    # reload config (or reuse) – assumes checkpoint_path is already correct in YAML
    segmenter_config = SegmenterConfig.from_yaml(str(config_folder_path / 'default_components' / external_method_config_name))

    aggregator_config = best_aggregator_config.model_copy(deep=True)

    tlm, rlm = test_benchmarker.benchmark(
        segmenter_config=segmenter_config,
        aggregator_config=aggregator_config,
        dataset_names=test_datasets,
    )

    # ---------- 3) Summaries ----------
    summary_tile = SegmenterBenchmarker.compute_mean_std_metric_tables(
        [tlm],
        output_path / 'tile_level_summary.csv',
    )

    summary_raster = SegmenterBenchmarker.compute_mean_std_metric_tables(
        [rlm],
        output_path / 'raster_level_summary.csv',
    )

    # ---------- 4) Combine tile & raster summaries ----------
    combined = SegmenterBenchmarker.merge_tile_and_raster_summaries(
        summary_tile,
        summary_raster,
        output_csv=output_path / 'combined_level_summary.csv',
        tile_prefix='tile',
        raster_prefix='raster',
    )



def main(exp_name, output_path_root, rf1_eval_iou_threshold, nms_algorithm):
    wandb_df_ood = None
    models_root_path_ood = '/network/scratch/h/xxx.xxx/training/detector_experience_OOD_datasets'
    models_root_path_multires = '/network/scratch/h/xxx.xxx/training/detector_experience_multi_resolution'
    models_root_path_singleres = '/network/scratch/h/xxx.xxx/training/detector_experience_resolution_optimalHPs_80m_FIXED'

    wandb_df_rebuttal = None
    wandb_df_segmenters = None

    models_root_path_rebuttal = ''
    models_root_path_segmenters = ''

    valid_nms_datasets = [
        #'SelvaBox',
        #'Detectree2',
        #'SelvaMask',
        'QuebecTrees'
    ]
    test_datasets = [
        # 'SelvaBox',
        #'BCI50ha',
        #'Detectree2',
       # 'OAM-TCD',
       # 'PanamaBCNM'
        # 'NeonTreeEvaluation',
        'QuebecTrees'
        # 'SelvaMask'
    ]

    raw_data_root = '/scratch/xxx/selvamask/datasets'


    exp_name_to_run_ids = {
        'NQOS': {
            'wandb_df': wandb_df_ood,
            'models_root': models_root_path_ood,
            'ids': [
                'dino_detrex_20250501_053202_618835_6696639',
                'dino_detrex_20250504_150959_967684_6720103',
                'dino_detrex_20250504_164639_096893_6720104'
            ]
        },
        'fasterrcnn_NQOS': {
            'wandb_df': wandb_df_rebuttal,
            'models_root': models_root_path_rebuttal,
            'ids':
            [
                'faster_rcnn_detectron2_20250726_003151_273358_7289182',
                'faster_rcnn_detectron2_20250727_162330_502343_7302074',
                'faster_rcnn_detectron2_20250727_162428_537731_7302075'
            ]
        },
        'maskrcnn_SelvaMask': {
            'wandb_df': wandb_df_segmenters,
            'models_root': models_root_path_segmenters,
            'ids':
            [
                'mask_rcnn_detectron2_20260107_035840_884189_8411559',
                'mask_rcnn_detectron2_20260109_010510_059578_8431761',
                'mask_rcnn_detectron2_20260109_012904_523020_8431762'
            ]
        },
        'mask2former_r50_SelvaMask': {
            'wandb_df': wandb_df_segmenters,
            'models_root': models_root_path_segmenters,
            'ids':
            [
                'mask2former_detrex_20260109_015312_079883_8431764',
                'mask2former_detrex_20260109_015312_071694_8431765',
                'mask2former_detrex_20260107_165617_105344_8416033'
            ]
        },
        'mask2former_swinL_SelvaMask': {
            'wandb_df': wandb_df_segmenters,
            'models_root': models_root_path_segmenters,
            'ids':
            [
                'mask2former_detrex_20260107_040138_848430_8411564',
                'mask2former_detrex_20260109_023308_032823_8431767',
                'mask2former_detrex_20260109_021141_650033_8431766'
            ]
        },
        'NQOS_SAM2': {
            'wandb_df': wandb_df_ood,
            'models_root': models_root_path_ood,
            'ids': [
                'dino_detrex_20250501_053202_618835_6696639',
                'dino_detrex_20250504_150959_967684_6720103',
                'dino_detrex_20250504_164639_096893_6720104'
            ],
            'segmenter_config': 'default_components/segmenter_sam2'
        },
        
        # --- NEW MANUAL EXPERIMENT ---
         'SAM2_Selvabox_frozen': {
            'wandb_df': None, 
            'models_root': '/data/xxx/selvamask/eval/selvabox_sam2',
            'ids': ['selvabox_frozen'], 
            'segmenter_config': 'default_components/segmenter_sam2',
            'detector_config': 'default_components/detector_multi_NQOS_best'
        },
    
        'SAM3_Selvabox_frozen': {
            'wandb_df': None, 
            'models_root': '/scratch/xxx/selvamask/eval/selvabox_sam3',
            'ids': ['selvabox_frozen'], 
            'segmenter_config': 'default_components/segmenter_sam3',
            'detector_config': 'default_components/detector_multi_NQOS_best'
        },
        
        'SAM2_Selvabox_ft': {#This needs to have multiple detector seeds
            'wandb_df': None, 
            'models_root': '/data/xxx/selvamask/eval/selvaboxft_sam2',
            'ids': ['frozen'], 
            'segmenter_config': 'default_components/segmenter_sam2',
            'detector_config': 'default_components/detector_selvaboxft',
        },
        'SAM2ft_Selvabox_ft': {#This needs to have multiple segmenter seeds
            'wandb_df': None, 
            'models_root': '/data/xxx/selvamask/eval/selvaboxft_sam2ft',
            'ids': ['seed42', 'seed1234', 'seed2024'], 
            'segmenter_config': 'default_components/segmenter_sam2',
            'detector_config': 'default_components/detector_selvaboxft',
            'seed_for' : 'segmenter'
        },
        'SAM3_Selvabox_ft': {#This needs to have multiple detector seeds
            'wandb_df': None, 
            'models_root': '/data/xxx/selvamask/eval/selvaboxft_sam3',
            'ids': ['frozen'], 
            'segmenter_config': 'default_components/segmenter_sam3',
            'detector_config': 'default_components/detector_selvaboxft'
        },
         'SAM3ft_Selvabox_ft': {#This needs to have multiple segmenter seeds
            'wandb_df': None, 
            'models_root': '/scratch/xxx/selvamask/eval/selvaboxft_sam3ft',
            'ids': ['seed42', 'seed1234', 'seed2024'], 
            'segmenter_config': 'default_components/segmenter_sam3',
            'detector_config': 'default_components/detector_selvaboxft',
            'seed_for' : 'segmenter'
        },
        'SAM3_Coco_ft': {#This needs to have multiple segmenter seeds
            'wandb_df': None, 
            'models_root': '/data/xxx/selvamask/eval/cocoft_sam3',
            'ids': ['frozen'], 
            'segmenter_config': 'default_components/segmenter_sam3',
            'detector_config': 'default_components/detector_cocoft'
        },
        'SAM3_Deepforest': {#This needs to have multiple segmenter seeds
            'wandb_df': None, 
            'models_root': '/scratch/xxx/selvamask/eval/deepforest_sam3',
            'ids': ['frozen'], 
            'segmenter_config': 'default_components/segmenter_sam3',
            'detector_config': 'default_components/detector_deepforest'
        },
        'Detectree2': {#This needs to have multiple segmenter seeds
            'wandb_df': None, 
            'models_root': '/scratch/xxx/selvamask/eval/detectree2',
            'ids': ['frozen']
        },
    }


    external_methods_configs = {
        'detectree2_flexi': 'detector_detectree2_flexi.yaml',
        'detectree2_randsizefull': 'detector_detectree2_randresizefull',
        'deepforest': 'detector_deepforest.yaml'
    }

    if rf1_eval_iou_threshold == "50:95":
        rf1_eval_iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if exp_name in exp_name_to_run_ids:
        exp_config = exp_name_to_run_ids[exp_name]

        output_path = f'{output_path_root}/{exp_name}'
        wandb_df = exp_config['wandb_df']
        run_ids = exp_config['ids']
        models_root_path = exp_config['models_root']
        seed_for = None
        if 'seed_for' in exp_config:
            seed_for = exp_config['seed_for']
        detector_config_path = None
        if 'detector_config' in exp_config and exp_config['detector_config'] is not None:
            detector_config_path = get_config_path(exp_config['detector_config'])
  
        if 'segmenter_config' in exp_config and exp_config['segmenter_config'] is not None:
            prompted_segmenter_config = exp_config['segmenter_config']
            if type(prompted_segmenter_config) is str:
                config_path = get_config_path(prompted_segmenter_config)
                prompted_segmenter_config = SegmenterConfig.from_yaml(config_path)
            
            if wandb_df is not None:
                df_sub = wandb_df[wandb_df['run_name'].isin(run_ids)]
                best_idx  = df_sub['bbox/AP.max'].idxmax()
                best_model_id = df_sub.at[best_idx, 'run_name']
            else:
                best_model_id = run_ids[0]

        else:
            prompted_segmenter_config = None
            if wandb_df is not None:
                df_sub = wandb_df[wandb_df['run_name'].isin(run_ids)]
                best_idx  = df_sub['segm/AP.max'].idxmax()
                best_model_id = df_sub.at[best_idx, 'run_name']
            else:
                best_model_id = run_ids[0]

        evaluate(
            run_ids=run_ids,
            best_run_id=best_model_id,
            models_root_path=models_root_path,
            raw_data_root=raw_data_root,
            output_path=output_path,
            valid_nms_datasets=valid_nms_datasets,
            test_datasets=test_datasets,
            rf1_eval_iou_threshold=rf1_eval_iou_threshold,
            nms_algorithm=nms_algorithm,
            prompted_segmenter_config=prompted_segmenter_config,
            detector_config_path=detector_config_path,
            seed_for=seed_for
        )
    elif exp_name in external_methods_configs:
        output_path = f'{output_path_root}/{exp_name}'

        evaluate_external_method(
            exp_name,
            external_methods_configs[exp_name],
            raw_data_root=raw_data_root,
            output_path=output_path,
            valid_nms_datasets=valid_nms_datasets,
            test_datasets=test_datasets,
            rf1_eval_iou_threshold=rf1_eval_iou_threshold,
            nms_algorithm=nms_algorithm,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a single experiment (NQ, NQO, NQOS or 30_120m)."
    )
    parser.add_argument(
        "exp_name",
        help="Experiment name to evaluate",
    )
    parser.add_argument(
        "--output_path_root",
        type=str,
        help="Root output path for evaluation results.",
    )
    parser.add_argument(
        "--rf1_eval_iou_threshold",
        help="IoU threshold to use for RF1 evaluation.",
    )
    parser.add_argument(
        "--nms_algorithm",
        help="NMS algorithm to use (e.g., iou, ioa-disambiguate, or other supported options).",
    )
    args = parser.parse_args()

    main(args.exp_name, args.output_path_root, args.rf1_eval_iou_threshold, args.nms_algorithm)
