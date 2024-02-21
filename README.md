# My ChRIS Plugin

[![Version](https://img.shields.io/docker/v/fnndsc/pl-TrainSeg?sort=semver)](https://hub.docker.com/r/fnndsc/pl-TrainSeg)
[![MIT License](https://img.shields.io/github/license/fnndsc/pl-TrainSeg)](https://github.com/FNNDSC/pl-TrainSeg/blob/main/LICENSE)
[![ci](https://github.com/FNNDSC/pl-TrainSeg/actions/workflows/ci.yml/badge.svg)](https://github.com/FNNDSC/pl-TrainSeg/actions/workflows/ci.yml)

`pl-TrainSeg` is a [_ChRIS_](https://chrisproject.org/)
_ds_ plugin which takes the train and valid data (.npy) as input files and
creates the weight for one view as output files.

## Abstract
In the fetal brain, the measurement of cortical thickness is sensitive to the segmentation of cortical plate (CP), because of the low resolution of magnetic resonance imaging (MRI) due to the relatively small brain size. 

High-resolution MRI data provides detailed delineation of CP enabling accurate cortical thickness measurement. 

This is the training of the Cortical Plate Segmentation in High Resolution MRIs and the second part of our complete pipeline; the input data will be the output of our first part `pl-HighPrepRes`.


## Installation

`pl-TrainSeg` is a _[ChRIS](https://chrisproject.org/) plugin_, meaning it can
run from either within _ChRIS_ or the command-line.

## Local Usage

To get started with local command-line usage, use [Apptainer](https://apptainer.org/)
(a.k.a. Singularity) to run `pl-TrainSeg` as a container:

```shell
apptainer exec docker://fnndsc/pl-TrainSeg SegTrain [--args values...] input/ output/
```

To print its available options, run:

```shell
apptainer exec docker://fnndsc/pl-TrainSeg SegTrain --help
```

## Background

`pl-TrainSeg` needs as input the directory with the folder with your data in numpy format `pl-HighPrepRes` and it reads all the input subdirs and runs a model trainer.

The output will be a folder with the three final weights in format .h5

## Examples

`SegTrain` requires two positional arguments: a directory containing
input data, and a directory where to create output data.

First, create the input directory and move input data into it (it must be a .npy data).

```shell
mkdir incoming/ outgoing/
mv some.dat other.dat incoming/
apptainer exec docker://fnndsc/pl-TrainSeg:latest SegTrain [--view] incoming/ outgoing/
```

## Development

Instructions for developers.

### Building

Build a local container image:

```shell
docker build -t localhost/fnndsc/pl-TrainSeg .
```

### Running

Mount the source code `SegTrain.py` into a container to try out changes without rebuild.

```shell
docker run --rm -it --userns=host -u $(id -u):$(id -g) \
    -v $PWD/SegTrain.py:/usr/local/lib/python3.11/site-packages/SegTrain.py:ro \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw -w /outgoing \
    localhost/fnndsc/pl-TrainSeg SegTrain /incoming /outgoing
```

### Testing

Run unit tests using `pytest`.
It's recommended to rebuild the image to ensure that sources are up-to-date.
Use the option `--build-arg extras_require=dev` to install extra dependencies for testing.

```shell
docker build -t localhost/fnndsc/pl-TrainSeg:dev --build-arg extras_require=dev .
docker run --rm -it localhost/fnndsc/pl-TrainSeg:dev pytest
```

## Release

Steps for release can be automated by [Github Actions](.github/workflows/ci.yml).
This section is about how to do those steps manually.

### Increase Version Number

Increase the version number in `setup.py` and commit this file.

### Push Container Image

Build and push an image tagged by the version. For example, for version `1.2.3`:

```
docker build -t docker.io/fnndsc/pl-TrainSeg:1.2.3 .
docker push docker.io/fnndsc/pl-TrainSeg:1.2.3
```

### Get JSON Representation

Run [`chris_plugin_info`](https://github.com/FNNDSC/chris_plugin#usage)
to produce a JSON description of this plugin, which can be uploaded to _ChRIS_.

```shell
docker run --rm docker.io/fnndsc/pl-TrainSeg:1.2.3 chris_plugin_info -d docker.io/fnndsc/pl-TrainSeg:1.2.3 > chris_plugin_info.json
```

Intructions on how to upload the plugin to _ChRIS_ can be found here:
https://chrisproject.org/docs/tutorials/upload_plugin

