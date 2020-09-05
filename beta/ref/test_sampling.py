import numpy as np
trials_num = 10000

def Thompson_sampling(trials, wins, p):
    p_beta = [np.random.beta(wins[i] + 1, trials[i] - wins[i] + 1) for i in range(len(trials))]
    choice = np.argmax(p_beta)
    
    trials[choice] += 1
    if p[choice] > np.random.rand():
        wins[choice] += 1

def UCB(trials, wins, p):
    mean = wins/trials
    sum = trials.sum()
    p_ucb = [mean[i] + np.sqrt(2*np.log(sum)/trials[i]) for i in range(len(trials))]
    choice = np.argmax(p_ucb)
    
    trials[choice] += 1
    if p[choice] > np.random.rand():
        wins[choice] += 1

def main():
    p = [0.1, 0.2, 0.3, 0.4, 0.5]
    trials = np.array([0, 0, 0, 0, 0])
    wins = np.array([0, 0, 0, 0, 0])

    for i in range(0, trials_num):
        Thompson_sampling(trials, wins, p)
    print('=======Thompson_sampling=====')
    print(p)
    print(trials)
    print(wins)
    print(wins/trials)
    

    trials = np.array([1, 1, 1, 1, 1])
    wins = np.array([1 if p[i] > np.random.rand() else 0 for i in range(len(p)) ])
    for i in range(len(p)):
        trials[i] += 1
        if p[i] > np.random.rand():
            wins[i] += 1
    for i in range(0, trials_num):
        UCB(trials, wins, p)
    print('=======UCB_sampling=====')
    print(trials)
    print(wins)
    print(wins/trials)
        

if __name__ == '__main__':
    main()
