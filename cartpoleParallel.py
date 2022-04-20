# pip3 install gym
# pip3 install neat-python

# for gym stuff: 
# apt install xvfb ffmpeg xorg-dev libsdl2-dev swig cmake
# pip3 install gym[box2d]

import multiprocessing
import os
import pickle

import neat
import numpy as np
#import cart_pole
import gym
import math
import os
import sys

sys.path.append('./neato')
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters



def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(4):
        env = gym.make("CartPole-v1")
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 1:
                #env.render()
            action = genome.forward(observation)[0]

            observation, reward, done, info = env.step(action <= 0.5)
            #print('observation: ', observation)
            #print("action ",action)
            #print("reward ", reward)
            fitness += reward
        #env.close()
        #print("fitness ", fitness)
        fitnesses.append(fitness)

    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    hyperparams.max_generations = 300

    brain = Brain(4, 1, 100, hyperparams)
    brain.generate()

    
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        current_best = brain.get_fittest()
        print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
            current_best.get_fitness() * 100, 
            current_gen, 
            hyperparams.max_generations
        ))


if __name__ == '__main__':
    run()
