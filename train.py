import argparse
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated"
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"shapely.set_operations"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"None of the inputs have requires_grad=True\. Gradients will be None"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*"
)

from canopyrs.engine.models.utils import set_all_seeds
from canopyrs.engine.utils import init_spawn_method

detrex_logger = logging.getLogger("detrex.checkpoint.c2_model_loading")
detrex_logger.disabled = True

from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.detector.train_detectron2.train_detectron2 import train_detectron2_fasterrcnn
from canopyrs.engine.models.detector.train_detectron2.train_detrex import train_detrex, eval_detrex
from canopyrs.engine.models.segmenter.train_sam.train_sam3 import train_sam3


def train_detector_main(args):
    config = DetectorConfig.from_yaml(args.config)

    if args.dataset:
        config.data_root_path = args.dataset

    if config.seed:
        set_all_seeds(config.seed)

    if config.model == 'faster_rcnn_detectron2':
        train_detectron2_fasterrcnn(config)
    if config.model == 'retinanet_detectron2':
        train_detectron2_fasterrcnn(config)
    elif config.model == 'dino_detrex':
        train_detrex(config)
    else:
        raise ValueError("Invalid model type/name.")

def train_segmenter_main(args):
    config = SegmenterConfig.from_yaml(args.config)

    if args.dataset:
        config.data_root_path = args.dataset

    if args.train_output_path:
        config.train_output_path = args.train_output_path
    
    if args.detector_config_path:
        config.detector_config_path = args.detector_config_path
    if args.detector_cache_dir:
        config.detector_cache_dir = args.detector_cache_dir
    if args.eval_pipeline_config_path:
        config.eval_pipeline_config_path = args.eval_pipeline_config_path
    if args.coco_eval_output_dir:
        config.coco_eval_output_dir = args.coco_eval_output_dir

    if args.seed is not None:
        config.seed = args.seed

    if config.seed:
        set_all_seeds(config.seed)

    if config.model == 'sam3':
        print("Starting SAM3 training...")
        train_sam3(config)

    else:
        raise ValueError(f"Invalid segmenter model: {config.model}")

def eval_detector_main(args):
    config = DetectorConfig.from_yaml(args.config)

    if args.dataset:
        config.data_root_path = args.dataset

    if config.seed:
        set_all_seeds(config.seed)

    if config.model == 'faster_rcnn_detectron2':
        raise NotImplementedError("Evaluation for FasterRCNN is not yet implemented.")
    elif config.model == 'dino_detrex':
        eval_detrex(config, fold_name=args.eval_only_fold)
    else:
        raise ValueError("Invalid model type/name.")


if __name__ == '__main__':
    init_spawn_method()
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument("-m", "--model", type=str, help="The type of model to train (detector, segmenter, classifier...).", required=True)
    parser.add_argument("-c", "--config", type=str, default='default', help="Name of a default, predefined config or path to the appropriate .yaml config file.")
    parser.add_argument("-d", "--dataset", type=str, help="Path to the root folder of the dataset to use for training a model. Will override whatever is in the yaml config file.")
    parser.add_argument("-o", "--train_output_path", type=str, help="Path to the output folder where to save model checkpoints and logs during training (only for segmenter training). Will override whatever is in the yaml config file.")
    parser.add_argument("--eval_only_fold", type=str, help="Whether to only evaluate the model. If defined, the value passed will be used as the fold name to find and load the appropriate data.")
    parser.add_argument("--detector_config_path", type=str, help="Path to detector config file to use for generating box prompts.")
    parser.add_argument("--detector_cache_dir", type=str, help="Path to cache detector outputs for prompts.")
    parser.add_argument("--eval_pipeline_config_path", type=str, help="Path to evaluation pipeline config file.")
    parser.add_argument("--coco_eval_output_dir", type=str, help="Path to save evaluation outputs (predictions, metrics, etc). Overrides yaml config.")
    parser.add_argument("--seed", type=int, help="Override random seed")
    args = parser.parse_args()

    if args.model == "detector":
        if args.eval_only_fold:
            eval_detector_main(args)
        else:
            train_detector_main(args)
    elif args.model == "segmenter":
        if args.eval_only_fold:
            raise NotImplementedError("Evaluation for segmenter models is not yet implemented.")
        else:
            train_segmenter_main(args)




