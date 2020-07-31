import gym
import time
env = gym.make("CartPole-v1")
observation = env.reset()

begin = time.time()
max_step = 300
for _ in range(max_step):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print (f'step is {_}', end='\r')
    if done:
        observation = env.reset()

during = time.time() - begin
print (f'during time : {round(during, 2)}s')
print (f'rate is {round(max_step/during, 2)}')

env.close()