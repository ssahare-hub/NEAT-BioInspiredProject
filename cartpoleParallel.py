# pip3 install gym
# for gym stuff: 
# apt install xvfb ffmpeg xorg-dev libsdl2-dev swig cmake
# pip3 install gym[box2d]



# To use render environment need to use only one cpu in brain and uncomment render part
#  or it will become frozen. (Probably way to fix this....)
import multiprocessing
import os
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



def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(4):
        env = gym.make("CartPole-v1")
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
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
    #hyperparams.max_generations = 300
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['edge'] = 0.05

    inputs = 4
    outputs = 1
    hidden_layers = 6
    population = 400
    
    brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
    brain.generate()

    
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        current_best = brain.get_fittest()
        print("Current Accuracy: {0} | Current species: {1} | Current genome: {2} | Current gen: {3}".format(
            current_best.get_fitness(), 
            brain.get_current_species()+1, 
            brain.get_current_genome()+1,
            current_gen
        ))


if __name__ == '__main__':
    run()