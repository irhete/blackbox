# encoding: utf-8
# cython: profile=True
# filename: train_weights.pyx

import interface as bbox
cimport interface as bbox
import numpy as np
from random import random, seed, randint
import math
from libc.math cimport exp

cdef:
    float last_score = 0.0, alpha = 0.0001, eps = 0.1, eps_decay = 1.05, alpha_decay = 1.05, gamma = 0.9
    int iterations = 1000, N = 0
    int n_features = 36, n_actions = 4, max_time = 1214494
    float w[4][36]
    float bias[4]
    float exp_states[1214494][36]
    int exp_actions[1214494]
    float exp_rewards[1214494]

output_file_w = "weights.npy"
output_file_bias = "bias.npy"

# sigmoid function
cdef float sigmoid(float x, int deriv=0):
    if(deriv):
        return x * (1 - x)
    return 1.0 / (1 + exp(-x))

def update_weights(int total):
    global w, bias
    cdef:
        int i, a, j, t
        float r, target = 0, delta
        float s[36]

    for t in range(N):
        i = N - t - 1
        
        s = exp_states[i]
        a = exp_actions[i]
        r = exp_rewards[i]

        target = r + gamma * target
        
        alpha = 1.0 / (total + t + 1)
        val = calc_val_for_action(s, a)
        delta = alpha * (target - val) * sigmoid(val, 1)
        for j in range(n_features):
            w[a][j] += delta * s[j]
        bias[a] += delta
    

cdef int get_action_by_state(float* state):
    cdef:
        int act, best_act = -1
        float val, best_val = -1e9
      
    if random() <= eps:
        return randint(0, n_actions-1)
    
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

    return sigmoid(val + bias[action])

 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(int total):
    global last_score, N, exp_states, exp_actions, exp_rewards
    
    cdef:
        int has_next = 1, action
        float current_score, reward
        float state[36]
    
    prepare_bbox()

    state = bbox.c_get_state()
    while has_next:
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)

        current_score = bbox.c_get_score()
        reward = current_score - last_score
        last_score = current_score

        new_state = bbox.c_get_state()

        # save experience  
        exp_states[N] = state
        exp_actions[N] = action
        exp_rewards[N] = reward

        state = new_state

        N += 1

        # batch updates
    update_weights(total)

    bbox.finish(verbose=1)


def train():
    global w, eps, N, bias, last_score, alpha
    cdef int i, j, total = 0
    for i in range(n_actions):
        for j in range(n_features):
            w[i][j] = np.random.rand()
        bias[i] = np.random.rand()
      
    for i in range(iterations):
        #eps = 1.0 / (iterations+1)
        N = 0
        last_score = 0.0
        run_bbox(total)
        eps /= eps_decay
        #alpha /= alpha_decay
        np.save(output_file_w, w)
        np.save(output_file_bias, bias)
        total += N
        print (i)
 
