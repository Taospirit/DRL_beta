import gym, os, time
import matplotlib.pyplot as plt
import torch

from drl.model import ActorDPG, CriticQ
from drl.policy import DDPGPolicy as Policy
from drl.buffer import ReplayBuffer as Buffer

env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]

hidden_dim = 32
episodes = 5000
max_step = 200
buffer_size = 50000
actor_learn_freq = 5
target_update_freq = 10
batch_size = 1000
model_save_dir = './save/test_ddpg'
os.makedirs(model_save_dir, exist_ok=True)

actor = ActorDPG(state_space, hidden_dim, action_space)
critic = CriticQ(state_space, hidden_dim, action_space)
buffer = Buffer(buffer_size)
policy = Policy(actor, critic, buffer, actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq, batch_size=batch_size)


def plot(steps, y_label, plot_save_dir):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.plot(steps)
    RunTime = len(steps)

    path = plot_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
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
run = run_type[1]
save_file = model_save_dir.split('/')[-1]

def main():
    if run == 'train':
        save_setting()
        live_time = []

        for i_eps in range(episodes):
            step = policy.sample(env, max_step)
            live_time.append(step)
            plot(live_time, 'Training_DDPG_TwoNet_Double', model_save_dir)
            loss_actor_avg, loss_critic_avg = policy.learn()
            print (f'Learing Eps:{i_eps}, Actor_Loss:{round(loss_actor_avg, 3)}, Critic_Loss:{round(loss_critic_avg, 3)}')

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
                action = policy.choose_action(state).clip(-1, 1)
                action *= action_max
                next_state, reward, done, info = env.step(action)
                rewards += reward
                env.render()
                if done:
                    state = env.reset()
                    break
                state = next_state
            print (f'Eval eps:{i+1}, Avg_reward:{rewards/(step+1)}, Steps:{step+1}')
        env.close()

    elif run == 'retrain':
        print ('Loading model...')
        policy.load_model(model_save_dir, save_file)
        live_time = []

        for i_eps in range(episodes):
            reward = policy.sample(env, max_step)
            live_time.append(reward)
            plot(live_time, 'Training_DDPG_TwoNet_Double', model_save_dir)
            policy.learn()
        policy.save_model(model_save_dir, save_file)
        env.close()

    else:
        print ('Setting your run type!')


if __name__ == '__main__':
    main()