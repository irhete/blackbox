import interface as bbox
import numpy as np

weights_file = "weights.npy"

def get_action_by_state(state, verbose=0):
    best_act = -1
    best_val = -1e9

    for act in range(n_actions):
        val = calc_action_val(act, state)
        if val > best_val:
			      best_val = val
			      best_act = act

    return best_act

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
    has_next = 1
    
    prepare_bbox()
 
    while has_next:
        state = bbox.get_state()
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)
 
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    w = np.load(weights_file)
    run_bbox(verbose=0)
 
