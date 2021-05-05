import os, sys, json, shutil, subprocess
import multiprocessing
import pickle

from metrics import *
from combination import *
from combination_with_estimation import *

def single_frame_string_result(ocrstring):
    temprover = AlignmentWithEstimation(0.6, ListBasedSequenceStructure)
    temprover.add_string(ocrstring)
    return temprover.get_string_result()

def convert_ocrstring(serialized_ocrstring):
    '''
    Converts a serialized text string recognition result to a list of Cells
    '''
    ret = []
    for serialized_ocrcell in serialized_ocrstring:
        varmap = serialized_ocrcell
        if '@' not in varmap.keys():
            varmap['@'] = 0.0
        ret.append(Cell(varmap))
    return ret

def clipmodel_full_combination(ocrstrings, foc_clip, recstep, gt, cache_dir):
    cache_file = os.path.join(cache_dir, 'clipmodel_full_combination_%d.pkl' % recstep)
    if os.path.exists(cache_file):
        cache = None
        with open(cache_file, 'rb') as ps:
            cache = pickle.load(ps)
        return cache
        
    rover = AlignmentWithEstimation(0.6, ListBasedSequenceStructure)
    
    ngld = [None for _ in ocrstrings]
    
    i_next_rec = 0
    for i_frame, ocrstring in enumerate(ocrstrings):
        if i_frame != i_next_rec: # skipping frame
            pass # not doing anything
        else: # recognizing frame
            rover.add_string(ocrstring)
            if i_frame + recstep - 1 < len(ocrstrings):
                ngld[i_frame + recstep - 1] = levmetric(rover.get_string_result(), gt)
            i_next_rec = i_frame + recstep
            
    curr_ngld = 1.0
    for i_frame in range(len(ngld)):
        if ngld[i_frame] is None:
            ngld[i_frame] = curr_ngld
        else:
            curr_ngld = ngld[i_frame]
    
    with open(cache_file, 'wb') as ps:
        pickle.dump(ngld, ps)
    return ngld

def clipmodel_1_best(ocrstrings, foc_clip, recstep, gt, cache_dir):
    cache_file = os.path.join(cache_dir, 'clipmodel_1_best_%d.pkl' % recstep)
    if os.path.exists(cache_file):
        cache = None
        with open(cache_file, 'rb') as ps:
            cache = pickle.load(ps)
        return cache
    
    best_focus_score = -1.0
    
    ngld = [None for _ in ocrstrings]
    
    i_next_rec = 0
    for i_frame, ocrstring in enumerate(ocrstrings):
        if i_frame != i_next_rec: # skipping frame
            pass # not doing anything
        else: # processing frame
            if foc_clip[i_frame] > best_focus_score:
                best_focus_score = foc_clip[i_frame]
                
                if i_frame + recstep - 1 < len(ocrstrings):
                    ngld[i_frame + recstep - 1] = levmetric(single_frame_string_result(ocrstring), gt)
                i_next_rec = i_frame + recstep
            else:
                i_next_rec = i_frame + 1
        
    curr_ngld = 1.0
    for i_frame in range(len(ngld)):
        if ngld[i_frame] is None:
            ngld[i_frame] = curr_ngld
        else:
            curr_ngld = ngld[i_frame]
    
    with open(cache_file, 'wb') as ps:
        pickle.dump(ngld, ps)
    return ngld

def get_best3_integration(ocrstrings, frames):
    sorted_frames = sorted([f for f in frames if f is not None])
    rover = AlignmentWithEstimation(0.6, ListBasedSequenceStructure)
    for f in sorted_frames:
        rover.add_string(ocrstrings[f])
    return rover.get_string_result()

def get_min_index(scores):
    i_min = 0
    for i in range(1, len(scores)):
        if scores[i] < scores[i_min]:
            i_min = i
    return i_min

def get_none_index(scores):
    for i in range(len(scores)):
        if scores[i] is None:
            return i
    return None

def clipmodel_3_best(ocrstrings, foc_clip, recstep, gt, cache_dir):
    cache_file = os.path.join(cache_dir, 'clipmodel_3_best_%d.pkl' % recstep)
    if os.path.exists(cache_file):
        cache = None
        with open(cache_file, 'rb') as ps:
            cache = pickle.load(ps)
        return cache
    
    best_focus_scores = [-1.0, -1.0, -1.0]
    best_focus_frames = [None, None, None]
        
    ngld = [None for _ in ocrstrings]
    
    i_next_rec = 0
    for i_frame, ocrstring in enumerate(ocrstrings):
        if i_frame != i_next_rec: # skipping frame
            pass # not doing anything
        else: # processing frame
            if foc_clip[i_frame] > min(best_focus_scores):
                i_min = get_min_index(best_focus_scores)
                best_focus_scores[i_min] = foc_clip[i_frame]
                best_focus_frames[i_min] = i_frame
                            
                if i_frame + recstep - 1 < len(ocrstrings):
                    ngld[i_frame + recstep - 1] = levmetric(get_best3_integration(ocrstrings, best_focus_frames), gt)
                i_next_rec = i_frame + recstep
            else:
                i_next_rec = i_frame + 1
        
    curr_ngld = 1.0
    for i_frame in range(len(ngld)):
        if ngld[i_frame] is None:
            ngld[i_frame] = curr_ngld
        else:
            curr_ngld = ngld[i_frame]
    
    with open(cache_file, 'wb') as ps:
        pickle.dump(ngld, ps)
    return ngld