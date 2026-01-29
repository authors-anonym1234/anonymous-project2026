# Selvamask: Anonymous Geospatial Tree Detection Pipeline

This repository contains the code for a pipeline designed to process high-resolution geospatial orthomosaics for the detection and segmentation of trees in various forest biomes.

## Features
- Detection and segmentation of trees in orthomosaic imagery
- Modular pipeline with configurable components
- Support for multiple detection and segmentation models

## Installation
Install the required Python packages in a python 3.10 conda environment:

```bash
conda create -n canopyrs python=3.10
conda activate canopyrs
conda install -c conda-forge gdal=3.6.2
git submodule update --init --recursive
python -m pip install -e .
python -m pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex
```

## ⚙️ Configuration

Each component is configurable via YAML configuration files.

While default configuration files are provided in the `canopyrs/config` directory,
you can also create your own configuration files by creating a new folder under `canopyrs/config/`, adding a `pipeline.yaml` script,
and setup your desired list of component configuration files.

A `pipeline` is made of multiple components, each with its own configuration. A typical `pipeline.yaml` configuration will look like this:

```yaml
components_configs:
  - tilerizer:
      tile_type: tile
      tile_size: 1777
      tile_overlap: 0.75
      ground_resolution: 0.045
  - detector: default_components/detector_multi_NQOS_best
  - aggregator:
      nms_algorithm: 'iou'
      score_threshold: 0.5
      nms_threshold: 0.7
      edge_band_buffer_percentage: 0.05
```

where `tilerizer`, `detector`, and `aggregator` are the names of the components, and `default_components/detector_multi_NQOS_best` points to a `[config_subfolder_name]/[component_name]` .yaml config in `canopyrs/config/`.

## 🚀 Inference

The main entry point of the inference pipeline is `infer.py`. 
This script accepts command-line arguments specifying the config to use and the input and output paths:

```bash
python infer.py -c <CONFIG_NAME> -i <INPUT_PATH> -o <OUTPUT_PATH>
```
Example run for a single raster/orthomosaic (`-i`) with a default config:
```bash
python infer.py -c default_detection_multi_NQOS_best -i /path/to/raster.tif -o <OUTPUT_PATH>
```

Example run for a folder of geo-referenced .tif images (`-t`) with a default config:
```bash
python infer.py -c default_detection_multi_NQOS_best -t /path/to/tiles/folder -o <OUTPUT_PATH>
```

## 🌳 Data

To download and extract datasets automatically and use it with our benchmark or training scripts, we provide a tool.

For example, to download SelvaBox and Detectree2 datasets, you can use the following command:

```bash
python -m tools.detection.download_datasets \
  -d SelvaBox Detectree2 \
  -o <DATA_ROOT>
```

## 📊 Evaluation

### Find optimal NMS parameters for Raster-level evaluation ($RF1_{75}$)
To find the optimal NMS parameters for your model, i.e. `nms_iou_threshold` and `nms_score_threshold`,
you can use the [`find_optimal_raster_nms.py`](canopyrs/tools/detection/find_optimal_raster_nms.py) tool script. This script will run a grid search over the NMS parameters and evaluate the results using the COCO evaluation metrics.
Depending on how many Rasters there are in the datasets you select, it could take from a few tens of minutes to a few hours. If you have lots of CPU cores, we recommend to increase the number of workers.

You have to pass the path of a detection model config file.

For example to find NMS parameters for the `default_detection_multi_NQOS_best` default model (DINO+Swin L-384 trained on NQOS datasets) on the validation set of SelvaBox and Detectree2 datasets,
you can use the following command (make sure to download the data first, see `Data` section):

```bash
python -m tools.detection.find_optimal_raster_nms \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --n_workers 6
```

For more information on parameters, you can use the `--help` flag:
```bash
python -m tools.detection.find_optimal_raster_nms --help
```


### Benchmarking
To benchmark a model on the test or valid sets of some datasets, you can use the [`benchmark.py`](canopyrs/tools/detection/benchmark.py) tool script.

This script will run the model and evaluate the results using COCO metrics (mAP and mAR).

If you provide `nms_threshold` and `score_threshold` parameters, it will also compute the $RF1_{75}$ metric by running an NMS at the raster level for datasets that have raster-level annotations.

For example, to benchmark the `default_detection_multi_NQOS_best` default model (DINO+Swin L-384 trained on NQOS datasets) on the test set of SelvaBox and Detectree2 datasets,
you can use the following command (make sure to download the data first, see `Data` section):

```bash
python -m tools.detection.benchmark \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --nms_threshold 0.7 \
  --score_threshold 0.5
```

By default the evaluation is done on the test set. 
For more information on parameters, you can use the `--help` flag:
```bash
python -m tools.detection.benchmark --help
```
## 🧠 Training

We provide a `train.py` script to train detector models on preprocessed datasets (you must download them first, see `Data`).

Currently, our training pipeline requires [wandb](https://wandb.ai/site) to be installed and configured for logging purposes.

Then, for example, if you want to train a model on the `SelvaBox` and `Detectree2` datasets, you will have to copy a `detector.yaml` config file, for example from [`config/default_detection_multi_NQOS_best`](canopyrs/config/default_detection_multi_NQOS_best/detector.yaml), and modify a few things:
- `model`: the model type, either `dino_detrex` for detrex-based DINO models or `faster_rcnn_detectron2` for detectron2-based Faster R-CNN models.
- `architecture`: the model architecture, either `dino-swin/dino_swin_large_384_5scale_36ep.py`, `dino-resnet/dino_r50_4scale_24ep.py` (for DINOs) or `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` (for Faster R-CNNs) are currently supported.
- `checkpoint_path`: path to the pretrained model checkpoint. You can keep the pretrained checkpoint we provide in order to fine tune it, or replace it with one of [detrex](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) COCO checkpoints.
- `data_root_path`: path to your dataset root folder (where the `SelvaBox` and `Detectree2` extracted datasets are, i.e. the <DATA_ROOT> folder in the `Data` section).
- `train_output_path`: path to the output folder where the model checkpoints and logs will be saved.
- `wandb_project`: name of the wandb project to log to (make sure to be [logged](https://wandb.ai/site)).
- `train_dataset_names`: A list of the names of the `location` folders (children of `data_root_path` you defined above) you want to train on. For example, `SelvaBox` has three locations, `brazil_zf2`, `ecuador_tiputini`, and `panama_aguasalud`. You can choose to train on all of them, or only on one or two of them. The same goes for `Detectree2` which has only one location, `malaysia_detectree2`.
- `valid_dataset_names`: A list of the names of the `location` folders (children of `data_root_path` you defined above) you want to validate on (see above for locations).

You can also modify plenty of other parameters such as `batch_size`, `lr`...

Finally, you can run the training script with the following command:

```bash
python train.py \
  -m detector \
  -c config/default_detection_multi_NQOS_best/detector.yaml 
```