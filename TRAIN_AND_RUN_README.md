# Step-by-step how to run the entire pipeline

## Table of Contents
- [Local installation](#localinstallation)
- [Docker installation](#dockerinstallation)
- [Data and pretrained models](#data)
- [Training](#training)
- [Docker local testing with pre-recorded data](#local)
- [Real-time](#realtime)

## Local Installation <a name = "localinstallation"></a>

Follow the following steps:
```
git clone git@github.com:PTG-Kitware/angel_system.git
git clone git@github.com:PTG-Kitware/TCN_HPL.git
git clone git@github.com:PTG-Kitware/yolov7.git
conda create --name angel_systen python=3.8.10
poetry lock --no-update
poetry install

```