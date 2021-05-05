import os, sys, json, shutil, subprocess
import multiprocessing
import pickle

from metrics import *
from combination import *
from combination_with_estimation import *

import math
import scipy.stats as stats

from model_basic import *

def rtmodel_full_combination(ocrstrings, foc_clip, recstep, gt, cache_dir):
    cache_file = os.path.join(cache_dir, 'rtmodel_full_combination_%d.pkl' % recstep)
    if os.path.exists(cache_file):
        cache = None
        with open(cache_file, 'rb') as ps:
            cache = pickle.load(ps)
        return cache
        
    bias = 0.1
    rover = AlignmentWithEstimation(0.6, ListBasedSequenceStructure)
    
    control_events = []
    
    num_processed_frames = 0
    i_next_rec = 0
    for i_frame, ocrstring in enumerate(ocrstrings):
        if i_frame != i_next_rec: # skipping frame
            pass # do nothing
        else: # recognizing frame
            rover.add_string(ocrstring)
            num_processed_frames += 1
            
            current_ngld = levmetric(rover.get_string_result(), gt)
            current_time_to_result = num_processed_frames * recstep
                        
            if num_processed_frames == 1: # setting delta as 1.0 always
                control_events.append((1.0, current_ngld, current_time_to_result))
            else:
                modelling_sum = rover.get_modelling_sum()
                delta = (bias + modelling_sum) / (num_processed_frames + 1)
                # registering event only if it's lower than the previous one
                # otherwise this event cannot happen
                if delta < control_events[-1][0]:
                    control_events.append((delta, current_ngld, current_time_to_result))

            i_next_rec = i_frame + recstep
            
    if control_events[-1][2] != num_processed_frames * recstep:
        control_events.append((-1.0, \
                               levmetric(rover.get_string_result(), gt), \
                               num_processed_frames * recstep))
    
    with open(cache_file, 'wb') as ps:
        pickle.dump(control_events, ps)
    return control_events

def prob_max_will_update(samples):
    if min(samples) == max(samples):
        return 1.0
    mean = sum(samples) / len(samples)
    sigma = math.sqrt(sum([(s - mean)**2 for s in samples]) / (len(samples) - 1))
    return 1.0 - stats.norm.cdf(max(samples), loc=mean, scale=sigma)

def single_frame_string_result(ocrstring):
    temprover = AlignmentWithEstimation(0.6, ListBasedSequenceStructure)
    temprover.add_string(ocrstring)
    return temprover.get_string_result()

def rtmodel_1_best(ocrstrings, foc_clip, recstep, gt, cache_dir):
    cache_file = os.path.join(cache_dir, 'rtmodel_1_best_%d.pkl' % recstep)
    if os.path.exists(cache_file):
        cache = None
        with open(cache_file, 'rb') as ps:
            cache = pickle.load(ps)
        return cache
           
    best_focus_score = -1.0
        
    control_events = []
    
    prev_best_result = None
    prev_delta = None
    last_p = None
    seen_focus_scores = []
    
    current_ngld = None
    num_skipped_frames = 0
    num_processed_frames = 0
    
    i_next_rec = 0
    for i_frame, ocrstring in enumerate(ocrstrings):
        if i_frame != i_next_rec: # skipping frame
            pass # not doing anything here
        else: # processing frame
            seen_focus_scores.append(foc_clip[i_frame])
            if foc_clip[i_frame] > best_focus_score:
                best_focus_score = foc_clip[i_frame]
                
                current_ngld = levmetric(single_frame_string_result(ocrstring), gt)
                num_processed_frames += 1
                
                if num_processed_frames == 1: # stopping after first recognition - start of the profile
                    control_events.append((100.0, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                    prev_delta = 100.0
                else:
                    last_p = levmetric_ocr(ocrstring, prev_best_result)
                    prob = prob_max_will_update(seen_focus_scores)
                    delta = last_p * prob / (prob * recstep + 1.0 - prob)
                    if delta < prev_delta:
                        control_events.append((delta, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                        prev_delta = delta
                
                prev_best_result = ocrstring
                i_next_rec = i_frame + recstep
            else:
                num_skipped_frames += 1
                if last_p is not None:
                    prob = prob_max_will_update(seen_focus_scores)
                    delta = last_p * prob / (prob * recstep + 1.0 - prob)
                    if delta < prev_delta:
                        control_events.append((delta, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                        prev_delta = delta
                i_next_rec = i_frame + 1
                
    control_events.append((-1.0, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                                      
    with open(cache_file, 'wb') as ps:
        pickle.dump(control_events, ps)
    return control_events

def get_best3_integration_rover(ocrstrings, frames):
    sorted_frames = sorted([f for f in frames if f is not None])
    rover = AlignmentWithEstimation(0.6, ListBasedSequenceStructure)
    for f in sorted_frames:
        rover.add_string(ocrstrings[f])
    return rover

def rtmodel_3_best(ocrstrings, foc_clip, recstep, gt, cache_dir):
    cache_file = os.path.join(cache_dir, 'rtmodel_3_best_%d.pkl' % recstep)
    if os.path.exists(cache_file):
        cache = None
        with open(cache_file, 'rb') as ps:
            cache = pickle.load(ps)
        return cache
           
    best_focus_scores = [-1.0, -1.0, -1.0]
    best_focus_frames = [None, None, None]
        
    control_events = []
    
    prev_best_result = None
    prev_delta = None
    last_p = None
    seen_focus_scores = []
    
    current_ngld = None
    num_skipped_frames = 0
    num_processed_frames = 0
    
    i_next_rec = 0
    for i_frame, ocrstring in enumerate(ocrstrings):
        if i_frame != i_next_rec: # skipping frame
            pass # not doing anything here
        else: # processing frame
            seen_focus_scores.append(foc_clip[i_frame])
            if foc_clip[i_frame] > min(best_focus_scores):
                i_min = get_min_index(best_focus_scores)
                best_focus_scores[i_min] = foc_clip[i_frame]
                best_focus_frames[i_min] = i_frame
                
                rover = get_best3_integration_rover(ocrstrings, best_focus_frames)
                current_ngld = levmetric(rover.get_string_result(), gt)
                num_processed_frames += 1
                
                if num_processed_frames == 1: # stopping after first recognition - start of the profile
                    control_events.append((100.0, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                    prev_delta = 100.0
                else:
                    last_p = levmetric_ocr(ocrstring, prev_best_result)
                    prob = prob_max_will_update(seen_focus_scores)
                    delta = last_p * prob / (prob * recstep + 1.0 - prob)
                    if delta < prev_delta:
                        control_events.append((delta, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                        prev_delta = delta
                
                prev_best_result = rover.base
                i_next_rec = i_frame + recstep
            else:
                num_skipped_frames += 1
                if last_p is not None:
                    prob = prob_max_will_update(seen_focus_scores)
                    delta = last_p * prob / (prob * recstep + 1.0 - prob)
                    if delta < prev_delta:
                        control_events.append((delta, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                        prev_delta = delta
                i_next_rec = i_frame + 1
                
    control_events.append((-1.0, current_ngld, num_processed_frames * recstep + num_skipped_frames))
                                      
    with open(cache_file, 'wb') as ps:
        pickle.dump(control_events, ps)
    return control_events
