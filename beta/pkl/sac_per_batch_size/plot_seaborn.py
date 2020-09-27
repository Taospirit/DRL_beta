import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

env_list = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Acrobot-v1', 'MountainCar-v0']
env_name = env_list[0]
file_path = os.path.abspath(os.path.dirname(__file__))
# file_path +='/sac2.pkl'
# alg = '/ddpg_'
alg = '/sac_per_'

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)

    return smooth_data


if __name__ == '__main__':
    # file = "behavior_cloning_" + ENV_NAME+".pkl"
    file = file_path + alg + env_name + '_batch_size_128' + '.pkl'
    with open(file, 'rb') as f:
        data1 = pickle.load(f)

    file = file_path + alg + env_name + '_batch_size_256' + '.pkl'
    with open(file, 'rb') as f:
        data2 = pickle.load(f)

    
    file = file_path + alg + env_name + '_batch_size_512' + '.pkl'
    with open(file, 'rb') as f:
        data3 = pickle.load(f)

    
    file = file_path + alg + env_name + '_batch_size_1024' + '.pkl'
    with open(file, 'rb') as f:
        data4 = pickle.load(f)
    
    file = file_path + alg + env_name + '_batch_size_2048' + '.pkl'
    with open(file, 'rb') as f:
        data5 = pickle.load(f)

    x1 = np.array(data1["mean"])[:, :1010]
    print (np.array(x1).shape)
    x1 = smooth(x1, sm=3)

    x2 = np.array(data2["mean"])[:, :1010]
    x2 = smooth(x2, sm=3)

    x3 = np.array(data3["mean"])[:, :1010]
    x3 = smooth(x2, sm=3)

    x4 = np.array(data4["mean"])[:, :1010]
    x4 = smooth(x2, sm=3)

    x5 = np.array(data5["mean"])[:, :1010]
    x5 = smooth(x5, sm=3)

    time = range(np.array(x1).shape[-1])

    sns.set(style="darkgrid", font_scale=1)
    sns.tsplot(time=time, data=x1, color="blue", condition="batch_size:128", linestyle='-')
    sns.tsplot(time=time, data=x2, color="b", condition="batch_size:256", linestyle='-')
    sns.tsplot(time=time, data=x3, color="r", condition="batch_size:512", linestyle='-')
    sns.tsplot(time=time, data=x4, color="g", condition="batch_size:1024", linestyle='-')
    sns.tsplot(time=time, data=x5, color="pink", condition="batch_size:2048", linestyle='-')

    plt.ylabel("Reward")
    plt.xlabel("Episodes Number")
    plt.title("SAC_Pendulum-v0")
    plt.savefig(file_path + '/test_batch_size.jpg')
    plt.show()