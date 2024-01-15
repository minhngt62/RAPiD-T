#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from scipy import signal, ndimage
import json
import time

from utils.repp_rotated_utils import get_video_frame_iterator, get_iou, get_pair_features
import cv2
import sys

INF = 9e15

# =============================================================================
# Robust and Efficient Post-Processing for Video Object Detection (REPP)
# =============================================================================


class REPP():
    
    def __init__(self, min_tubelet_score, add_unmatched, min_pred_score,
              distance_func, clf_thr, clf_mode,
              recoordinate, recoordinate_std, model_path,
              annotations_filename = '',
              **kwargs):
        
        self.min_tubelet_score = min_tubelet_score      # threshold to filter out low-scoring tubelets
        self.min_pred_score = min_pred_score            # threshold to filter out low-scoring base predictions
        self.add_unmatched = add_unmatched              # True to add unlinked detections to the final set of detections. Leads to a lower mAP

        self.distance_func = distance_func              # LogReg to use the learning-based linking model. 'def' to use the baseline from SBM
        self.clf_thr = clf_thr                          # threshold to filter out detection linkings
        self.clf_mode = clf_mode                        # Relation between the logreg score and the semmantic similarity. 'dot' recommended

        self.recoordinate = recoordinate                # True to perform a recordinating step
        self.recoordinate_std = recoordinate_std        # Strength of the recoordinating step
        self.model_path = model_path                    # Path of the bounding box matching model

        if self.distance_func == 'logreg':
            print('Loading clf matching model:', self.model_path)
            self.clf_match, self.matching_feats = pickle.load(open(self.model_path, 'rb'))
            self.match_func = self.distance_logreg
        else: raise ValueError('distance_func not recognized:', self.distance_func)
    
        

    def distance_def(self, p1, p2):
        iou = get_iou(p1['bbox'][:], p2['bbox'][:])
        score = np.dot(p1['scores'], p2['scores'])
        div = iou * score
        if div == 0: return INF
        return 1 / div
    
    # Computes de linking score between a pair of detections
    def distance_logreg(self, p1, p2):
        pair_features = get_pair_features(p1, p2, self.matching_feats)          #, image_size[0], image_size[1]
        score = self.clf_match.predict_proba(np.array([[ pair_features[col] for col in self.matching_feats ]]))[:,1]
        if score < self.clf_thr: return INF

        if self.clf_mode == 'max': 
            score = p1['scores'].max() * p2['scores'].max() * score
        elif self.clf_mode == 'dot':
            score = np.dot(p1['scores'], p2['scores']) * score
        elif self.clf_mode == 'dot_plus':
            score = np.dot(p1['scores'], p2['scores']) + score
        elif self.clf_mode == 'def':
            return distance_def(p1, p2)
        elif self.clf_mode == 'raw':
            pass
        else: raise ValueError('error post_clf')
        return 1 - score
        

    # Return a list of pairs of frames lnked accross frames
    def get_video_pairs(self, preds_frame):
        num_frames = len(preds_frame)
        frames = list(preds_frame.keys())
        frames = sorted(frames)
        
        pairs, unmatched_pairs = [], []
        for i in range(num_frames - 1):
            
            pairs_i = []
            frame_1, frame_2 = frames[i], frames[i+1]
            preds_frame_1, preds_frame_2 = preds_frame[frame_1], preds_frame[frame_2]
            num_preds_1, num_preds_2 = len(preds_frame_1), len(preds_frame_2)
            
            # Any frame has no preds -> save empty pairs
            if num_preds_1 != 0 and num_preds_2 !=  0: 
                # Get distance matrix
                distances = np.zeros((num_preds_1, num_preds_2))
                for i,p1 in enumerate(preds_frame_1):
                    for j,p2 in enumerate(preds_frame_2):
                        distances[i,j] = self.match_func(p1, p2)
                
                # Get frame pairs
                pairs_i = self.solve_distances_def(distances, maximization_problem=False)
                
            unmatched_pairs_i = [ i for i in range(num_preds_1) if i not in [ p[0] for p in pairs_i] ]
            pairs.append(pairs_i); unmatched_pairs.append(unmatched_pairs_i)
     
        return pairs, unmatched_pairs

    # Solve distance matrix and return a list of pair of linked detections from two consecutive frames
    def solve_distances_def(self, distances, maximization_problem):
        pairs = []
        if maximization_problem:
            while distances.min() != -1:
                inds = np.where(distances == distances.max())
                a,b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
                a,b = int(a), int(b)
                pairs.append((a, b))
                distances[a,:] = -1
                distances[:,b] = -1
        else:
            while distances.min() != INF:
                inds = np.where(distances == distances.min())
                a,b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
                a,b = int(a), int(b)
                pairs.append((a, b))
                distances[a,:] = INF
                distances[:,b] = INF
    
        return pairs        


    # Create tubelets from list of linked pairs
    def get_tubelets(self, preds_frame, pairs):
    
        num_frames = len(preds_frame)
        frames = list(preds_frame.keys())
        tubelets, tubelets_count = [], 0
        
        first_frame = 0
        
        
        while first_frame != num_frames-1:
            ind = None    
            for current_frame in range(first_frame, num_frames-1):
                
                # Continue tubelet
                if ind is not None:
                    pair = [ p for p in pairs[current_frame] if p[0] == ind ]
                    # Tubelet ended
                    if len(pair) == 0:
                        tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                        tubelets_count += 1
                        ind = None
                        break            
                    
                    # Continue tubelet
                    else:
                        pair = pair[0]; del pairs[current_frame][pairs[current_frame].index(pair)]
                        tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                        ind = pair[1]
                        
                # Looking for a new tubelet
                else:
                    # No more candidates in current frame -> keep searching
                    if len(pairs[current_frame]) == 0: 
                        first_frame = current_frame+1
                        continue
                    # Beginning a new tubelet in current frame
                    else:
                        pair = pairs[current_frame][0]; del pairs[current_frame][0]
                        tubelets.append([(current_frame, 
                              preds_frame[frames[current_frame]][pair[0]])])
                        ind = pair[1]
        
            # Tubelet has finished in the last frame
            if ind != None:
                tubelets[tubelets_count].append((current_frame+1, preds_frame[frames[current_frame+1]][ind])) # 4
                tubelets_count += 1
                ind = None    
                
        return tubelets
        

    # Performs the re-scoring refinment
    def rescore_tubelets(self, tubelets, avg_func="mean"):    
        for t_num in range(len(tubelets)):
            t_scores = [ p['scores'] for _,p in tubelets[t_num] ]
            if avg_func == "mean" or avg_func == "mean_plus":
                new_scores = np.mean(t_scores, axis=0)
            elif avg_func == "max":
                new_scores = np.amax(t_scores, axis=0)
            for i in range(len(tubelets[t_num])): 
                if avg_func == "mean_plus":
                    tubelets[t_num][i][1]['scores'] = np.maximum(new_scores, tubelets[t_num][i][1]['scores'])
                else:
                    tubelets[t_num][i][1]['scores'] = new_scores
            
            for i in range(len(tubelets[t_num])):
                if 'emb' in tubelets[t_num][i][1]: del tubelets[t_num][i][1]['emb']
            
        return tubelets
    
    
    # Performs de re-coordinating refinment
    def recoordinate_tubelets_full(self, tubelets, ms=-1):
        
        if ms == -1: ms = 40
        for t_num in range(len(tubelets)):
            t_coords = np.array([ p['bbox'] for _,p in tubelets[t_num] ])
            w = signal.gaussian(len(t_coords)*2-1, std=self.recoordinate_std*100/ms)
            w /= sum(w)
            
            for num_coord in range(5):
                t_coords[:,num_coord] = ndimage.convolve(t_coords[:,num_coord], w, mode='reflect')
                
            for num_bbox in range(len(tubelets[t_num])): 
                tubelets[t_num][num_bbox][1]['bbox'] = t_coords[num_bbox,:].tolist()
                
        return tubelets
    
    
    # Extracts predictions from tubelets
    def tubelets_to_predictions(self, tubelets_video, preds_format):
        
        preds, track_id_num = [], 0
        for tub in tubelets_video:
            for _,pred in tub:
                    for cat_id, s in enumerate(pred['scores']):
                        if s < self.min_pred_score: continue
                        if preds_format == 'cepdof':
                            bbox = list(map(float, pred['bbox']))
                            bbox[0] += bbox[2]/2; bbox[1] += bbox[3]/2
                            preds.append({
                                    # 'image_id': pred['image_id'].split('/')[-1].split('.')[0],
                                    'image_id': pred['image_id'],
                                    'bbox': bbox,
                                    'score': float(s),
                                    'segmentation':[],
                                    'track_id': track_id_num,
                                })
                        else: raise ValueError('Predictions format not recognized')
            track_id_num += 1
        return preds
    
    
    def __call__(self, video_predictions, postproc=True):
        # Filter out low-score predictions
        for frame in video_predictions.keys():
            video_predictions[frame] = [ p for p in video_predictions[frame] if max(p['scores']) >= self.min_tubelet_score ]
        
        video_predictions = dict(sorted(video_predictions.items()))
        
        pairs, unmatched_pairs = self.get_video_pairs(video_predictions)

        tubelets = self.get_tubelets(video_predictions, pairs)
        
        if postproc:
            tubelets = self.rescore_tubelets(tubelets, avg_func="mean")
            if self.recoordinate: tubelets = self.recoordinate_tubelets_full(tubelets)
        
        if self.add_unmatched:
            print('Adding unmatched')
            tubelets += self.add_unmatched_pairs_as_single_tubelet(unmatched_pairs, video_predictions)
        
        predictions_cepdof = self.tubelets_to_predictions(tubelets, 'cepdof')

        return predictions_cepdof




def det_arr_to_det_dict(
    preds_arr,
    root_dir_frames="/projectnb/arpae/mtezcan/datasets/fisheye/youtube/frames_cropped",
    ):
    im2wh = {}
    im2bboxes = {}
    for k, bbox_obj in enumerate(preds_arr):

        image_id = bbox_obj['image_id']
        if not image_id in im2bboxes:
            im2bboxes[image_id] = []
        x, y, w, h, angle = bbox_obj["bbox"]
        x1, y1 = x - w/2, y - h/2

        if not image_id in im2wh:
            try:
                im = cv2.imread(f"{root_dir_frames}/{image_id}.png")[:, :, ::-1]
            except:
                continue

            im2wh[image_id] = [im.shape[1], im.shape[0]]

        im_w, im_h = im2wh[image_id]
        cx, cy = x / im_w, y / im_h

        im2bboxes[image_id].append({
            "image_id":image_id,
            "bbox": [x1, y1, w, h, angle],
            "bbox_center": [cx, cy],
            "scores": [bbox_obj["score"]],
        })

    return im2bboxes