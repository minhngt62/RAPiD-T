**Work In Progress**

# RAPiD-T: Leveraging Temporal Information for People Detection from Overhead Fisheye Cameras

**This repository is the original implemetation RAPiD-REPP, RAPiD-FA and RAPiD-FGFA introduced in the following paper. It is forked from [original repository of RAPiD](https://github.com/duanzhiihao/RAPiD.)**

## Requirements
* Python 3.8.10
* pytorch 1.9.0
* opencv 4.5.0

## Inference Instructions
* You can donwloade the weights of trained models from (Google Drive)[https://drive.google.com/drive/folders/1G66FOZT4gY56cw63twANtS_Tqf3j5AtO?usp=sharing]

* Following code is tested with the example frames that can be downloaded from (Google Drive)[https://drive.google.com/file/d/1zcJcx1sOPD015sHpWy9OHVUlxXj2owFT/view?usp=sharing]. You need unzip this file and put `warehouse_samples` in `examples`.

* If you want to use your own dataset, please make sure that your frames are named as `<video_name>.<6 digit frameid>.png`.

### RAPiD+REPP
TBD
### RAPiD+FA and RAPiD+FGFA
Follow the steps in `inference/RAPiD-FA.ipynb` and `inference/RAPiD-FGFA.ipynb` to compute the detections and produce a video with detections shown on top of the frames.

