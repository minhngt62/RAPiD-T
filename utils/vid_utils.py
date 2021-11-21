from PIL import Image
import cv2
import numpy as np
import glob
import os
import json
from utils import visualization
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from utils.flow_utils import warp_feat
import torchvision.transforms.functional as tvf
from utils import rotated_bboxes as rbb

def saveDets(
    detector, 
    vid_folder, 
    det_folder, 
    softmax_T=10,
    frame_diff=5, 
    img_ext=".png", 
    conf_thres=0.05, 
    flow_subsample = 8,
    use_nms=True,
    input_size=1024,
    flow_warping=False,
):

    torch.cuda.reset_peak_memory_stats()
    print(f"Confidence Threshold is {conf_thres}")
    frame_paths = sorted(glob.glob(os.path.join(vid_folder, f"*{img_ext}")))
    if not os.path.exists(det_folder):
        os.makedirs(det_folder)

    feats_past = []
    feats_future = []
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    tensor_softmax = nn.Softmax(dim=0)

    for k in range(min(frame_diff, len(frame_paths))):
        frame = Image.open(frame_paths[k])
        small, medium, large, pad_info, img_size = detector.predict_pil_feats(
            frame, 
            input_size=input_size,
            test_aug=None,
            )
        feats_future.append([frame, small, medium, large, pad_info, k])

    for k, frame_path in enumerate(frame_paths):
        # Compute detections
        assert len(feats_future) > 0
        feats_k = feats_future.pop(0)
        frame_k, small_k, medium_k, large_k, pad_info_k, k_ = feats_k
        # print(k, k_)
        assert k == k_

        if k + frame_diff < len(frame_paths):
            frame = Image.open(frame_paths[k + frame_diff])
            small, medium, large, pad_info, img_size = detector.predict_pil_feats(
                frame, 
                input_size=input_size,
                test_aug=None,
                )
            feats_future.append([frame, small, medium, large, pad_info, k + frame_diff])
        
        # Compute feature embeddings
        small_k_feat = detector.model(small_k, embedding_out="small")
        medium_k_feat = detector.model(medium_k, embedding_out="medium")
        large_k_feat = detector.model(large_k, embedding_out="large")

        small_arr, medium_arr, large_arr = [], [], []
        small_w, medium_w, large_w = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()
        
        for frame, small, medium, large, _, k2 in [*feats_past, feats_k, *feats_future]:          

            if flow_warping and k != k2:
                frame_k_np = np.array(frame_k).astype(np.uint8)
                frame_np = np.array(frame).astype(np.uint8)
                h, w, c = frame_np.shape

                frame_k_np = cv2.resize(frame_k_np, (int(w / flow_subsample), int(h / flow_subsample)))
                frame_np = cv2.resize(frame_np, (int(w / flow_subsample), int(h / flow_subsample)))

                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY), 
                    cv2.cvtColor(frame_k_np, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    
                flow = torch.unsqueeze(tvf.to_tensor(flow).type(torch.FloatTensor), 0).cuda()
                small = warp_feat(small, flow)
                medium = warp_feat(medium, flow)
                large = warp_feat(large, flow)
                del flow

            small_arr.append(small)
            medium_arr.append(medium)
            large_arr.append(large)

            small_feat = detector.model(small, embedding_out="small")
            medium_feat = detector.model(medium, embedding_out="medium")
            large_feat = detector.model(large, embedding_out="large")
            small_w = torch.cat((small_w, softmax_T * torch.unsqueeze(cos_sim(small_feat[0], small_k_feat[0]), 0)))
            medium_w = torch.cat((medium_w, softmax_T * torch.unsqueeze(cos_sim(medium_feat[0], medium_k_feat[0]), 0)))
            large_w = torch.cat((large_w, softmax_T * torch.unsqueeze(cos_sim(large_feat[0], large_k_feat[0]), 0)))

        small_w = tensor_softmax(small_w)
        medium_w = tensor_softmax(medium_w)
        large_w = tensor_softmax(large_w)

        for l in range(len(small_arr)):
            small_arr[l] = small_arr[l] * small_w[l]
            medium_arr[l] = medium_arr[l] * medium_w[l]
            large_arr[l] = large_arr[l] * large_w[l]

        small_mean = torch.sum(torch.stack(small_arr), dim=0)
        medium_mean = torch.sum(torch.stack(medium_arr), dim=0)
        large_mean = torch.sum(torch.stack(large_arr), dim=0)
        detections = detector.predict_pil_dets(
            small_mean, 
            medium_mean, 
            large_mean, 
            pad_info_k,
            return_img=False,
            input_size=input_size,
            conf_thres=conf_thres,
            test_aug=None,
            use_nms=use_nms,
            img_size=img_size,
        )
        
        feats_past.append(feats_k)
        if len(feats_past) > frame_diff:
            feats_past.pop(0)

        frame_name = frame_paths[k].split('/')[-1].split('.')[0]
        np.savetxt(os.path.join(
            det_folder, f"{frame_name}.txt"), detections, fmt="%.8f")


def txt2json(det_folder, json_path):
    detections = []
    det_paths = sorted(glob.glob(os.path.join(det_folder, "*.txt")))
    for det_path in det_paths:
        image_id = det_path.split('/')[-1].split('.')[0]
        dets = np.loadtxt(det_path)
        for det in dets.reshape(-1, 6):
            x, y, w, h, a, c = det
            detections.append({
                "image_id": image_id,
                "bbox": [x, y, w, h, a],
                "score": c,
                "segmentation": []
            })

    with open(json_path, 'w') as outfile:
        json.dump(detections, outfile, indent=4)



def json2dict(json_path, include_ids=False):
    with open(json_path) as f:
        detections = json.load(f)

    if "annotations" in detections:
        detections = detections["annotations"]

    im2dets = {}
    for det in detections:
        image_id = det['image_id']
        bbox = det['bbox'].copy()
        score = det['score'] if 'score' in det else 0.0
        bbox.append(score)
        if include_ids:
            assert 'person_id' in det or 'track_id' in det
            if 'person_id' in det:
                bbox.append(det['person_id'])
            elif 'track_id' in det:
                bbox.append(det['track_id'])
        bbox = np.asarray(bbox)
        if image_id in im2dets:
            im2dets[image_id].append(bbox)
        else:
            im2dets[image_id] = [bbox]

    return im2dets


def saveVideo(vid_folder, im2boxes_arr, out_path, img_ext=".png", show_count=True, show_id=False, fps=30, out_size=None):
    frames = set()
    for im2boxes, _, _, _ in im2boxes_arr:
        frames = frames.union(set(im2boxes.keys()))
    frames = sorted(list(frames))
    if out_size is None:
        image_path = os.path.join(vid_folder, f"{frames[0]}{img_ext}")
        img = Image.open(image_path)
        np_img = np.array(img)
        h, w, _ = np_img.shape
    else:
        w, h = out_size
        ratio = h / w

    vid = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))

    for frame in frames:
        image_path = os.path.join(vid_folder, f"{frame}{img_ext}")
        if not os.path.exists(image_path):
            print(f"{image_path} does not exist")
            continue
        
        img = Image.open(image_path)
        np_img = np.array(img)
        for im2boxes, color, show_conf, conf_thres in im2boxes_arr:
            if frame in im2boxes:
                bbox = im2boxes[frame]
                visualization.draw_dt_on_np(
                    np_img, bbox, conf_thres=conf_thres, color=color, show_conf=show_conf,
                    show_count=show_count, show_id=show_id)
        
        if out_size is not None:
            h_im, w_im, c_im = np_img.shape
            if h_im < (w_im * ratio):
                pad = int(((w_im * ratio)- h_im) / 2)
                np_img = cv2.copyMakeBorder(np_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
            if w_im < (h_im / ratio):
                pad = int(((h_im / ratio) - w_im) / 2)
                np_img = cv2.copyMakeBorder(np_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        vid.write(cv2.resize(np_img[:, :, ::-1], (w, h)))

    vid.release()   