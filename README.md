**Work In Progress**

# RAPiD-T: Leveraging Temporal Information for People Detection from Overhead Fisheye Cameras

**This repository is the original implemetation RAPiD-REPP, RAPiD-FA and RAPiD-FGFA introduced in the following paper.**

## Disclaimer
* RAPiD-T is implemeted by combining [RAPiD](http://openaccess.thecvf.com/content_CVPRW_2020/html/w38/Duan_RAPiD_Rotation-Aware_People_Detection_in_Overhead_Fisheye_Images_CVPRW_2020_paper.html) with the SOTA people detections algorithms designed for side-view regular cameras, namely [REPP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9341600) and [FGFA](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.html).
* This repository is forked from [original repository of RAPiD](https://github.com/duanzhiihao/RAPiD).
* For REPP, we used the [official implementation](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) and adapted it to rotated bounding boxes.
* For FA and FGFA, we implemented our own versions based on our based understanding. 


## Requirements
* Python 3.8.10
* pytorch 1.9.0
* opencv 4.5.0

## Inference Instructions
* You can donwload the weights of trained models from [Google Drive](https://drive.google.com/drive/folders/1G66FOZT4gY56cw63twANtS_Tqf3j5AtO?usp=sharing). Place these weight files in `weights` folder.

* Following code is tested with the example frames that can be downloaded from [Google Drive](https://drive.google.com/file/d/1zcJcx1sOPD015sHpWy9OHVUlxXj2owFT/view?usp=sharing). You need to unzip this file and put `warehouse_samples` in `examples`.

* If you want to use your own dataset, please make sure that your frames are named as `<video_name>.<6 digit frameid>.png`.

* Follow the steps in `inference/RAPiD-REPP.ipynb`, `inference/RAPiD-FA.ipynb` and `inference/RAPiD-FGFA.ipynb` to compute the detections and produce a video with detections shown on top of the frames.

