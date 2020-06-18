# o-----------A
import torch
import numpy as np
import time

n_states = 11
actions = ['left', 'right']
epsilon = 0.9  # 贪心策略，90%概率选择最优动作，10%选择随机动作
alpha = 0.1  # learning rate
LAMBDA = 0.9  # 对未奖励的衰减值
MAX_EPISODES = 23  # 最大训练次数
FRESH_TIME = 0.3  # 走一步花的时间，为了更清楚的看到走的步骤

def build_q_table(n_states, n_actions):
    # 全0初始化Q_table
    q_table = torch.zeros(n_states, n_actions)
    return q_table

def choose_action(state, q_table):
    '''
    根据当前的状态和q table里面的值选择动作
    :param state:
    :param q_table:
    :return:
    '''
    state_actions = q_table[state, :]
    random_action = np.random.uniform(0, 1)
    if (random_action > epsilon) or (state_actions.sum().item() == 0):
        action_name = np.random.choice(actions)
    else:
        action_name = actions[state_actions.argmax().item()]
    return action_name

def update_environment(s, episode, step_counter):
    env_list = ['-'] * (n_states - 1) + ['T']
    if s == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('{}'.format(interaction))
        time.sleep(0.2)
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.2)

def get_env_feed_back(s, a):
    if a == 'right':
        s_ = s+1
        if s_ == n_states - 1:
            s_ = 'terminal'
            r = 1
        else:
            r = 0
    else:
        r = 0
        if s == 0:
            s_ = s
        else:
            s_ = s - 1
    return s_, r

def reinforcement_learning():
    q_table = build_q_table(n_states, len(actions))
    for episode in range(MAX_EPISODES):
        step_counter = 0  # 计数器，用于统计此轮训练工走了 多少步
        s = 0  # 初试状态位置
        is_terminal = False
        while not is_terminal:
            a = choose_action(s, q_table)  # 选择一个行为
            s_, r = get_env_feed_back(s, a)
            q_predict = q_table[s, actions.index(a)]  # q_table中的预测值
            if s_ != 'terminal':
                q_target = r + LAMBDA * q_table[s_, :].max().item()
            else:
                q_target = r
                is_terminal = True
            q_table[s, actions.index(a)] += alpha * (q_target - q_predict)
            s = s_  # 移动到下一个状态
            update_environment(s, episode, step_counter + 1)
            step_counter += 1
    return q_table

reinforcement_learning()
