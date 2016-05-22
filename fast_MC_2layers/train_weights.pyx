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
    int n_features = 36, n_actions = 4, max_time = 1214494, hidden_layer = 20
    float w[4][36][20]
    float h[4][20]
    float bias1[4][20]
    float bias2[4]
    float exp_states[1214494][36]
    int exp_actions[1214494]
    float exp_rewards[1214494]
    int best_action_batch_size = 100

output_file_w = "weights.npy"
output_file_bias1 = "bias1.npy"
output_file_h = "hidden_layer.npy"
output_file_bias2 = "bias2.npy"

# sigmoid function
cdef float sigmoid(float x, int deriv=0):
    if(deriv):
        return x * (1 - x)
    return 1.0 / (1 + exp(-x))


def update_weights(int total):
    global w, bias1, h, bias2
    cdef:
        int i, a, j, t, k
        float r, target = 0
        float s[36]
        float l1[20]
        float l2
        float l2_delta
        float l1_delta[20]

    for t in range(N):
        i = N - t - 1
        
        s = exp_states[i]
        a = exp_actions[i]
        r = exp_rewards[i]

        target = r + gamma * target
        
        alpha = 1.0 / (total + t + 1)
        
        # forwardprop
        l1 = calc_l1_for_action(s, a)
        l2 = calc_l2_for_action(l1, a)

        # backprop
        l2_delta = (target - l2) * sigmoid(l2, 1)
        l1_delta = calc_l1_delta(l2_delta, a, l1)

        for j in range(hidden_layer):
            h[a][j] += alpha * l1[j] * l2_delta
        bias2[a] += alpha * l2_delta
        
        for k in range(hidden_layer):
            for j in range(n_features):
                w[a][j][k] += alpha * l1_delta[k] * s[j]
            bias1[a][k] += alpha * l1_delta[k]

    

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

cdef int calc_best_action_using_checkpoint():
    # Pretty straightforward — we create a checkpoint and get it's ID 
    checkpoint_id = bbox.create_checkpoint()

    cdef int best_action = -1, action
    cdef float best_score = -1e9

    for action in range(n_actions):
        for _ in range(best_action_batch_size):
            bbox.do_action(action)

        if bbox.c_get_score() > best_score:
            best_score = bbox.c_get_score()
            best_action = action

        bbox.load_from_checkpoint(checkpoint_id)

    return best_action


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


cdef float* calc_l1_delta(float l2_delta, int action, float* l1):
    cdef:
        int j
        float l1_delta[20]

    for j in range(hidden_layer):
        l1_delta[j] = l2_delta * h[action][j] * sigmoid(l1[j], 1)

    return l1_delta


cdef float calc_l2_for_action(float* hidden_val, int action):
    cdef:
        int j
        float val = 0

    for j in range(hidden_layer):
        val += sigmoid(hidden_val[j]) * h[action][j]

    return sigmoid(val + bias2[action])


cdef float calc_val_for_action(float* s, int a):
    l1 = calc_l1_for_action(s, a)
    l2 = calc_l2_for_action(l1, a)

 
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
        action = calc_best_action_using_checkpoint() #get_action_by_state(state)
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
    np.random.seed(22)
    global w, eps, N, bias, last_score, alpha
    cdef int i, j, total = 0, a
    for a in range(n_actions):
        for j in range(hidden_layer):
            for i in range(n_features):
              w[a][i][j] = np.random.rand()
            bias1[a][j] = np.random.rand()
    
    for a in range(n_actions):
        for j in range(hidden_layer):
            h[a][j] = np.random.rand()
        bias2[a] = np.random.rand()

      
    for i in range(iterations):
        #eps = 1.0 / (iterations+1)
        N = 0
        last_score = 0.0
        run_bbox(total)
        eps /= eps_decay
        #alpha /= alpha_decay
        np.save(output_file_w, w)
        np.save(output_file_bias1, bias1)
        np.save(output_file_h, h)
        np.save(output_file_bias2, bias2)
        total += N
        print (i)
 
