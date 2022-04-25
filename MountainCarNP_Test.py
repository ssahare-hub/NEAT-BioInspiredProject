import os
import pickle
import neat
import gym 
import numpy as np
import math

with open('winner', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'configMountainCar')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)


env = gym.make("MountainCarContinuous-v0") 
for _ in range(5):
    last_observation = env.reset()

    fitness = 0.0
    done = False
    while not done:
        action = net.activate(last_observation)

        next_observation, reward, done, info = env.step(action)
        reward = reward = 100*((math.sin(3*next_observation[0]) * 0.0025 + 0.5 * next_observation[1] * next_observation[1]) - (math.sin(3*last_observation[0]) * 0.0025 + 0.5 * last_observation[1] * last_observation[1]))
        last_observation = next_observation
        fitness += reward
        env.render()
    print(fitness)