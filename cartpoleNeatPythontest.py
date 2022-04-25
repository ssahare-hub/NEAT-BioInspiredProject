import os
import pickle
import neat
import gym 
import numpy as np

# load the winner
with open('cartpoleNeatPythonBest', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'cartpoleNeatPythonconfig')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)


env = gym.make("CartPole-v1")
env._max_episode_steps = 750
observation = env.reset()

done = False
while not done:
    action = net.activate(observation)[0]

    observation, reward, done, info = env.step(action<= 0.5)
    env.render()