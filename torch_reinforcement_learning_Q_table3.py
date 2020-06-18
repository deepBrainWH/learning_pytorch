#0 0 0
#0 0 0
#0 0 1
import torch
import numpy as np

states = 8
actions = ['up', 'down', 'left', 'right']
alpha = 0.1
beita = 0.9
epoches = 20

def build_environment():
    env = torch.zeros(3,3)
    env[8] = 1
    return env

def build_q_table(n_states, n_actions):
    q_table = torch.zeros(n_states, n_actions)
    return q_table

def get_environment_feed_back(current_state, action, env):
    '''
    :param current_state: Tuple元组，表示位置信息
    :param action: 采取的行为
    :return:
    '''
    i = actions.index(action)
    if i == 0:
        current_state[0] -= 1
    elif i == 1:
        current_state[0] += 1
    elif i == 2:
        current_state[1] -= 1
    else:
        current_state[1] += 1
    r = env[current_state[0], current_state[1]]
    return current_state, r

def choose_action(current_state, q_table):
    i = current_state[0] * 3 + current_state[1]
    if q_table[i].sum().item() == 0:
        random_choice = np.random.choice(actions)
        action_index = actions.index(random_choice)
    else:
        action_index = q_table[i].argmax().item()
    action = actions[action_index]
    if action == 'up':
        current_state[0] = current_state[0] - 1 if current_state[0]>0 else current_state[0]

def reinforcement_learning():
    q_table = build_q_table(states, len(actions))
    for i in range(epoches):
        step_count = 0
        is_end = False
        curren_state = [0,0]
        while not is_end:
            a = choose_action(curren_state, q_table)


