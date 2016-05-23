import interface as bbox
cimport interface as bbox
from sys import argv
import numpy as np
from libc.math cimport exp

cdef:
    int n_features = 36, n_actions = 4, hidden_layer = 20
    float w[4][36][20]
    float h[4][20]
    float bias1[4][20]
    float bias2[4]

# sigmoid function
cdef float sigmoid(float x, int deriv=0):
    if(deriv):
        return x * (1 - x)
    return 1.0 / (1 + exp(-x))

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

cdef float calc_val_for_action(float* s, int a):
    l1 = calc_l1_for_action(s, a)
    l2 = calc_l2_for_action(l1, a)

cdef float* calc_l1_for_action(float* state, int action):
    cdef:
        int i, j
        float hidden_val[20]

    for j in range(hidden_layer):
        hidden_val[j] = bias1[action][j]

    for i in range(n_features):
        for j in range(hidden_layer):
            hidden_val[j] += state[i] * w[action][i][j]

    for j in range(hidden_layer):
        hidden_val[j] = sigmoid(hidden_val[j])
        
    return hidden_val

cdef float calc_l2_for_action(float* hidden_val, int action):
    cdef:
        int j
        float val = 0

    for j in range(hidden_layer):
        val += sigmoid(hidden_val[j]) * h[action][j]

    return sigmoid(val + bias2[action])
 
 
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
    w_pyth = np.load("weights.npy")
    bias1_pyth = np.load("bias1.npy")
    h_pyth = np.load("hidden_layer.npy")
    bias2_pyth = np.load("bias2.npy")

    for a in range(n_actions):
        for j in range(hidden_layer):
            for i in range(n_features):
              w[a][i][j] = w_pyth[a][i][j]
            bias1[a][j] = bias1_pyth[a][j]
    
    for a in range(n_actions):
        for j in range(hidden_layer):
            h[a][j] = h_pyth[a][j]
        bias2[a] = bias2_pyth[a]

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
