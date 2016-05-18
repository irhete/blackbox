# encoding: utf-8
# cython: profile=True
# filename: train_weights.pyx

import interface as bbox
cimport interface as bbox
import numpy as np
from random import random, seed, randint
import math

cdef:
    float last_score = 0.0, alpha = 0.0002, eps = 0.25, eps_decay = 1.05, alpha_decay = 1.05, gamma = 0.9
    int iterations = 1000, batch_size = 7, N = 0
    int n_features = 36, n_actions = 4, max_time = 1214494,
    float w[4][36]
    float bias[36]
    float exp_states[1214494][36]
    float exp_new_states[1214494][36]
    int exp_actions[1214494]
    int exp_has_next[1214494]
    float exp_rewards[1214494]

output_file_w = "weights.npy"
output_file_bias = "bias.npy"

def update_weights():
    global w, bias
    cdef:
        int i, a, smpl_has_next, a_prim, j
        float r, target, delta
        float val, best_val

    for _ in range(batch_size):
        best_val = -1e9
        i = randint(0, N)
        s = exp_states[i]
        a = exp_actions[i]
        r = exp_rewards[i]
        s_prim = exp_new_states[i]
        smpl_has_next = exp_has_next[i]

        if not smpl_has_next:
            target = r
        else:
            if random() <= eps:
                best_val = calc_val_for_action(s_prim, randint(0, n_actions-1))
            else:
                for a_prim in range(n_actions):
                    val = calc_val_for_action(s_prim, a_prim)
                    if val > best_val:
                        best_val = val
            target = r + gamma * best_val

        delta = alpha * (target - calc_val_for_action(s, a))
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

    return val + bias[action]

 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox():
    global w, last_score, N, exp_states, exp_new_states, exp_actions, exp_has_next, exp_rewards
    
    cdef:
        int has_next = 1, action
        float current_score, reward
        float state[36]
        float new_state[36]
    
    prepare_bbox()

    state = bbox.c_get_state()
    while has_next:
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)

        current_score = bbox.c_get_score()
        reward = current_score - last_score
        last_score = current_score

        new_state = bbox.c_get_state()
  
        exp_states[N] = state
        exp_new_states[N] = new_state
        exp_actions[N] = action
        exp_has_next[N] = has_next
        exp_rewards[N] = reward

        update_weights()
        N += 1

        state = new_state

    bbox.finish(verbose=1)


def train():
    global w, eps, N, bias, last_score, alpha
    cdef int i, j
    for j in range(n_features):
        for i in range(n_actions):
            w[i][j] = np.random.rand()
        bias[j] = np.random.rand()
      
    for i in range(iterations):
        N = 0
        last_score = 0.0
        run_bbox()
        eps /= eps_decay
        alpha /= alpha_decay
        np.save(output_file_w, w)
        np.save(output_file_bias, bias)
        print (i)
 
