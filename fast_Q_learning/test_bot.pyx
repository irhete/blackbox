import interface as bbox
cimport interface as bbox
from sys import argv
import numpy as np

cdef:
    float w[4][36]
    float bias[36]

cdef int get_action_by_state(float* state):
    cdef:
        int act, best_act = -1
        float val, best_val = -1e9
      
    for act in range(n_actions):
        val = calc_val_for_action(state, act)
        if val > best_val:
            best_val = val
            best_act = act
    
    return best_act


cdef float calc_val_for_action(float* state, int action):
    cdef:
        int i
        float val = 0

    for i in range(n_features):
        val += state[i] * w[action][i]

    return val + bias[action]
 
 
def prepare_bbox(level_type):
    global n_features, n_actions
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/%s_level.data"%level_type, verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
 
def run_bbox():
    global w, bias
    w = np.load("weights.npy")
    bias = np.load("bias.npy")
    level_type = "train" if len(argv) < 2 else argv[1]

    cdef:
        float state[36]
        int action, has_next = 1
    
    prepare_bbox(level_type)
 
    while has_next:
        state = bbox.c_get_state()
        action = get_action_by_state(state)
        has_next = bbox.c_do_action(action)
 
    bbox.finish(verbose=1)
