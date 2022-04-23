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
from neato.hyperparameters import Hyperparameters, tanh



def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(4):
        env = gym.make("Pendulum-v1")
        env._max_episode_steps = 750
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
            action = genome.forward(observation)[0]

            observation, reward, done, info = env.step(action*2)
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
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.max_fitness = 0

    inputs = 3
    outputs = 1
    hidden_layers = 6
    population = 400
    
    brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
    brain.generate()
    print(hyperparams.max_fitness)

    
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

    with open('best_genome', 'wb') as f:
        pickle.dump(current_best, f)

if __name__ == '__main__':
    run()
