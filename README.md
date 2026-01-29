# Selvamask

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


To evaluate a trained model or experiment, use the evaluation script with the desired experiment name and options. For example:

```bash
python evaluate.py <EXP_NAME> \
  --output_path_root /path/to/output_directory \
  --rf1_eval_iou_threshold "50:95" \
  --nms_algorithm ioa-disambiguate
```

Replace `<EXP_NAME>`, output path, and other arguments as needed for your setup.

By default the evaluation is done on the test set. 
For more information on parameters, you can use the `--help` flag:
```bash
python -m tools.detection.benchmark --help
```
## 🧠 Training

We provide a `train.py` script to train detector models on preprocessed datasets (you must download them first, see `Data`).

Currently, our training pipeline requires [wandb](https://wandb.ai/site) to be installed and configured for logging purposes.

Then, for example, if you want to train a SAM3 segmenter model, you will need to specify the appropriate configuration file, output directory, and data root. For instance:

For example, to train a SAM3 segmenter model, you can run:

```bash
python train.py -m segmenter \
  --config canopyrs/config/train_segmenter/segmenter_sam3_3.yaml \
  -o /path/to/output_directory \
  -d /path/to/data_root
```

Replace the config path, output directory, and data root with your own as needed.
