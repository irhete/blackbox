import interface as bbox
import numpy as np
from random import random, seed, randint
import math

last_score = 0.0
alpha = 0.001
eps = 0.18
eps_decay = 1.01
gamma = 0.9
iterations = 1000
batch_size = 10
output_file = "weights.npy"
w = None
experiences = []

def update_weights():
    global w

    for _ in range(batch_size):
        s, a, r, s_prim, smpl_has_next = experiences[randint(0, len(experiences)-1)]
        if not smpl_has_next:
          target = r
        else:
          target = r + gamma * max([np.dot(s_prim, w[a_prim]) for a_prim in range(n_actions)])
        delta_w = alpha * (target - np.dot(s,w[a])) * s
        # update w
        w[a] += delta_w
    

def get_action_by_state(state, verbose=0):
    if random() <= eps:
        return randint(0, n_actions-1)
    vals = [calc_action_val(act, state) for act in xrange(n_actions)]

    return np.argmax(vals)

def calc_action_val(action, state):
    return np.dot(state, w[action])


n_features = n_actions = max_time = -1
 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(verbose=False):
    global w, last_score, experiences
    has_next = 1
    
    prepare_bbox()

    if w is None:
        w = np.random.rand(n_actions, n_features)
  
    while has_next:
        state = bbox.get_state()
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)

        current_score = bbox.get_score()
        reward = current_score - last_score
        last_score = current_score

        new_state = bbox.get_state()

        experiences.append((state, action, reward, new_state, has_next))

        update_weights()

    bbox.finish(verbose=1)
    
 
 
if __name__ == "__main__":
    for i in range(iterations):
      run_bbox(verbose=0)
      eps /= eps_decay
      np.save(output_file, w)
      print (i)
 
