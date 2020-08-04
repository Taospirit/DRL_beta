import gym, os, time
import matplotlib.pyplot as plt
import torch

from drl.model import ActorNet, CriticV
from drl.policy import A2CPolicy as Policy

env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

hidden_dim = 32
episodes = 300
max_step = 1000
actor_learn_freq=1
target_update_freq=0
model_save_dir = './save/test_a2c'
os.makedirs(model_save_dir, exist_ok=True)

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticV(state_space, hidden_dim, 1)
policy = Policy(actor, critic, actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq)


def plot(steps, y_label, model_save_dir):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Reward')
    ax.plot(steps)
    RunTime = len(steps)
    path = model_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 50 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)

def save_setting():
    line = '===============================\n'
    env_info = f'env: {env_name} \nstate_space: {state_space}, action_space: {action_space}\n' 
    env_info += f'episodes: {episodes}, max_step: {max_step}\n'
    policy_dict = vars(policy)
    policy_info = ''
    for item in policy_dict.keys():
        policy_info += f'{item}: {policy_dict[item]} \n'

    data = line.join([env_info, policy_info])

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = dir_path + model_save_dir[1:] + '/' + model_save_dir.split('/')[-1] + '.txt'
    with open(path, 'w+') as f:
        f.write(data)

run_type = ['train', 'eval', 'retrain']
run = run_type[0]
save_file = model_save_dir.split('/')[-1]

def main():
    if run == 'train':
        save_setting()
        live_time = []
        for i_eps in range(episodes):
            step = policy.sample(env, max_step)
            live_time.append(step)
            plot(live_time, 'Training_A2C_TwoNet_no_Double', model_save_dir)
            policy.learn()

        policy.save_model(model_save_dir, save_file)
        env.close()

    elif run == 'eval':
        print ('Loading model...')
        policy.load_model(model_save_dir, save_file, test=True)

        for i in range(100):
            # reward = policy.sample(env, max_step, test=True)
            # print (f'eval episode:{i+1}, reward_avg:{reward}')
            state = env.reset()
            rewards = 0
            for step in range(max_step):
                action = policy.choose_action(state)
                next_state, reward, done, info = env.step(action)
                rewards += reward
                env.render()
                if done:
                    state = env.reset()
                    break
                state = next_state
            print (f'Eval eps:{i+1}, Rewards:{rewards}, Steps:{step+1}')
        env.close()

    elif run == 'retrain':
        print ('Loading model...')
        policy.load_model(model_save_dir, save_file)
        live_time = []

        for i_eps in range(episodes):
            reward = policy.sample(env, max_step)
            live_time.append(reward)
            plot(live_time, 'Training_A2C_TwoNet_no_Double', model_save_dir)
            policy.learn()
        policy.save_model(model_save_dir, save_file)
        env.close()

    else:
        print ('Setting your run type!')


if __name__ == '__main__':
    main()