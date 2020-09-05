import gym, os, time
import matplotlib.pyplot as plt
import torch

from drl.model import ActorDPG, CriticQ
from drl.algorithm import DDPG

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
actor_learn_freq = 1
target_update_freq = 0
batch_size = 1000

model_save_dir = 'save/test_ddpg'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

actor = ActorDPG(state_space, hidden_dim, action_space)
critic = CriticQ(state_space, hidden_dim, action_space)
# buffer = Buffer(buffer_size)
policy = DDPG(actor, critic, buffer_size=buffer_size, actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq, batch_size=batch_size)

def sample(env, policy, max_step, test=False):
    assert env, 'You must set env for sample'
    reward_avg = 0
    state = env.reset()

    for step in range(max_step):
        action = policy.choose_action(state, test)
        action *= env.action_space.high[0]
        next_state, reward, done, info = env.step([action])
        env.render()
        # process env callback
        if not test:
            policy.process(s=state, a=action, s_=next_state, r=reward)
        
        reward_avg += reward
        if done:
            break
        state = next_state

    if not test:
        policy.learn()
    return reward_avg/(step+1), step

from test_tool import policy_test
run_type = ['train', 'eval', 'retrain']
run = run_type[1]
plot_name = 'Training_DDPG_TwoNet_no_Double'

def main():
    test = False
    if run == 'eval':
        global episodes
        episodes = 100
        test = True
        print ('Loading model...')
        policy.load_model(model_save_dir, save_file, load_actor=True)

    elif run == 'train': 
        print ('Saving model setting...')
        # save_setting()
        # woc, 这句写的太丑了
        policy_test.save_setting(env_name, state_space, action_space, episodes, max_step, policy, model_save_dir, save_file)
    elif run == 'retrain':
        print ('Loading model...')
        policy.load_model(model_save_dir, save_file, load_actor=True, load_critic=True)
    else:
        print ('Setting your run type!')
        return 0

    live_time = []
    for i_eps in range(episodes):
        rewards, step = sample(env, policy, max_step, test=test)
        if run == 'eval':
            print (f'Eval eps:{i_eps+1}, Rewards:{rewards}, Steps:{step+1}')
            continue
        live_time.append(rewards)
        policy_test.plot(live_time, plot_name, model_save_dir)

        if i_eps > 0 and i_eps % 100 == 0:
            print (f'i_eps is {i_eps}')
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    env.close()

if __name__ == '__main__':
    main()