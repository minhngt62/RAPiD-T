# RAPiD-T: Leveraging Temporal Information for People Detection from Overhead Fisheye Cameras

**This repository is the original implemetation RAPiD-REPP, RAPiD-FA and RAPiD-FGFA. Corresponding paper will be published in [WACV 2022](https://wacv2022.thecvf.com/)**

* RAPiD-T is implemeted by combining [RAPiD](http://openaccess.thecvf.com/content_CVPRW_2020/html/w38/Duan_RAPiD_Rotation-Aware_People_Detection_in_Overhead_Fisheye_Images_CVPRW_2020_paper.html) with the SOTA object tracking algorithms designed for side-view regular cameras, namely [REPP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9341600) and [FGFA](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.html).
* This repository is forked from [original repository of RAPiD](https://github.com/duanzhiihao/RAPiD).
* For REPP, we used the [official implementation](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) and adapted it to rotated bounding boxes.
* For FA and FGFA, we implemented our own versions based on our based understanding. 


## Requirements
* [Python 3.8.10](https://www.python.org/downloads/release/python-3810/)
* [pytorch 1.9.0](https://pytorch.org/get-started/locally/)
* [opencv 4.5.0](https://opencv.org/opencv-4-5-0/)

## Inference Instructions
* You can donwload the weights of trained models from [Google Drive](https://drive.google.com/drive/folders/1G66FOZT4gY56cw63twANtS_Tqf3j5AtO?usp=sharing). Place these weight files in `weights` folder.

* Following code is tested with the example frames that can be downloaded from [Google Drive](https://drive.google.com/file/d/1zcJcx1sOPD015sHpWy9OHVUlxXj2owFT/view?usp=sharing). You need to unzip this file and put `warehouse_samples` in `examples`.

* If you want to use your own dataset, please make sure that your frames are named as `<video_name>.<6 digit frameid>.png`.

* Follow the steps in `inference/RAPiD-REPP.ipynb`, `inference/RAPiD-FA.ipynb` and `inference/RAPiD-FGFA.ipynb` to compute the detections and produce a video with detections shown on top of the frames.

## Citation

RAPiD-T source code is available for non-commercial use. If you find our code and dataset useful or publish any work reporting results using this source code, please consider citing our paper

```
 M.O. Tezcan, Z. Duan, M. Cokbas, P. Ishwar, and J. Konrad, “WEPDTOF: A Dataset and Benchmark 
 Algorithms for In-the-Wild People Detection and Tracking from Overhead Fisheye Cameras” 
 in Proc. IEEE/CVF Winter Conf. on Applications of Computer Vision (WACV), 2022.
 ```
