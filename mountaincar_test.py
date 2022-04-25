import pickle

import numpy as np
#import cart_pole
import gym
import math
import os
import sys

sys.path.append('./neato')
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters



def run():
    with open('mountaincar_best_individual', 'rb') as f:
        genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].showConn()

    env = gym.make("MountainCarContinuous-v0") # BipedalWalker-v3
    for _ in range(5):
        done = False
        observation = env.reset()
        while not done:
            action = genome.forward(observation)
            observation, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    run()