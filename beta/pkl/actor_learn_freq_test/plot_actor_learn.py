import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

env_list = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Acrobot-v1', 'MountainCar-v0']
env_name = env_list[0]
file_path = os.path.abspath(os.path.dirname(__file__))
# file_path +='/sac2.pkl'

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)

    return smooth_data

if __name__ == '__main__':
    # name = 'actor_learn_freq_'
    name = '_learn_freq_'
    file = file_path + '/sac_per_' + env_name + name + '1.pkl'
    with open(file, 'rb') as f:
        data1 = pickle.load(f)

    file = file_path + '/sac_per_' + env_name + name + '3.pkl'
    with open(file, 'rb') as f:
        data2 = pickle.load(f)

    file = file_path + '/sac_per_' + env_name + name + '5.pkl'
    with open(file, 'rb') as f:
        data3 = pickle.load(f)

    file = file_path + '/sac_per_' + env_name + name + '7.pkl'
    with open(file, 'rb') as f:
        data4 = pickle.load(f)

    file = file_path + '/sac_per_' + env_name + name + '9.pkl'
    with open(file, 'rb') as f:
        data5 = pickle.load(f)

    # print (np.array(x1).shape)
    x1 = smooth(data1["mean"], sm=5)
    x2 = smooth(data2['mean'], sm=5)
    x3 = smooth(data3['mean'], sm=5)
    x4 = smooth(data4['mean'], sm=5)
    x5 = smooth(data5['mean'], sm=5)

    time = range(np.array(x1).shape[-1])

    sns.set(style="darkgrid", font_scale=1)
    sns.tsplot(time=time, data=x1, color="blue", condition="actor_freq:1", linestyle='-')
    sns.tsplot(time=time, data=x2, color="b", condition="actor_freq:3", linestyle='-')
    sns.tsplot(time=time, data=x3, color="r", condition="actor_freq:5", linestyle='-')
    sns.tsplot(time=time, data=x4, color="g", condition="actor_freq:7", linestyle='-')
    sns.tsplot(time=time, data=x5, color="pink", condition="actor_freq:9", linestyle='-')

    plt.ylabel("Reward")
    plt.xlabel("Episodes Number")
    plt.title("SAC_Pendulum-v0")

    save_name = '/actor_learn_freq.jpg'
    save_path = file_path + save_name
    # print (save_path)
    plt.savefig(save_path)
    plt.show()