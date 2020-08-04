import numpy as np

def gae(rewards, v_evals, gamma, lam): # [r1, r2, ..., rT], [V1, V2, ... ,VT, VT+1]
    # GAE(gamma, lam=0) = td_error
    # GAE(gamma, lam=1) = MC
    assert len(v_evals) == len(rewards) + 1, 'V_pred length must one more than rewards length'
    rew_len = len(rewards)
    adv_gae = np.empty(rew_len, 'float32')
    lastgaelam = 0
    for i in reversed(range(rew_len)):
        nonterminal = 1 # to be fixed
        delta = rewards[i] + gamma * v_evals[i+1] * nonterminal - v_evals[i]
        adv_gae[i] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        # print (f'at index{i}, adv is {}')
    # return adv_gae
    ret = [v + adv for v, adv in zip(v_evals, adv_gae)]
    print (f'age_ret {ret}')

def compute_returns(rewards, masks, gamma=0.99):
    assert len(rewards) == len(masks), 'rewards & masks must have same length'
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        # print (masks[-2:])
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    # return returns
    print (f'com_ret {returns}')

r = [1, 2, 5, 7, 11, -20, 1, 2, 4]
v = [1, 2, 3, 4, 5, 6, 1, 2, 3]
m = [1, 1, 1, 1, 1, 0, 1, 1, 1]
v.append(0)
compute_returns(r, m, gamma=0.1)
gae(r, v, gamma=0.1, lam=1)


