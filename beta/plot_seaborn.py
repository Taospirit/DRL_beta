import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



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
    # file = "behavior_cloning_" + ENV_NAME+".pkl"
    # with open(os.path.join("test_data", file), "rb") as f:
    with open(file_path + '/sac.pkl', 'rb') as f:
        data = pickle.load(f)

    with open(file_path + '/sac_per.pkl', 'rb') as f:
        data2 = pickle.load(f)

    x1 = data["mean"]
    print (np.array(x1).shape)
    x1 = smooth(x1, sm=5)

    x2 = data2['mean']
    x2 = smooth(x2, sm=5)

    time = range(np.array(x1).shape[-1])

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(time=time, data=x1, color="r", condition="SAC", linestyle='-')
    sns.tsplot(time=time, data=x2, color="b", condition="SAC_pre", linestyle='-')

    plt.ylabel("Reward")
    plt.xlabel("Episodes Number")
    plt.title("Pendulum-v0")

    plt.show()