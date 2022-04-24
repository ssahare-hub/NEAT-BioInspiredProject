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
from neato.hyperparameters import Hyperparameters, tanh, sigmoid

EPISODE_DURATION = 500
seed = 0 # environment seed

def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(2):
        env = gym.make("BipedalWalker-v3")
        env.reset(seed=seed)
        #env._max_episode_steps = EPISODE_DURATION
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
            action = genome.forward(observation)
            #print(action)
            observation, reward, done, info = env.step(action)
            fitness += reward
        #env.close()
        #print("fitness ", fitness)
        fitnesses.append(fitness)
    sys.stdout.flush()
    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 1.5
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.75
    hyperparams.fitness_offset = 100 #*EPISODE_DURATION
    #hyperparams.max_fitness = hyperparams.fitness_offset
    #hyperparams.max_generations = 30

    inputs = 24
    outputs = 4
    hidden_layers = 2
    population = 20
    
    brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
    brain.generate()
    
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        brain.update_fittest()
        current_best = brain.get_fittest()
        print("Current Accuracy: {0} | Current species: {1} | Current genome: {2} | Current gen: {3}".format(
            current_best.get_fitness(), 
            brain.get_current_species()+1, 
            brain.get_current_genome()+1,
            current_gen
        ))

    with open('bipedal_best_individual', 'wb') as f:
        pickle.dump(current_best, f)

if __name__ == '__main__':
    run()
