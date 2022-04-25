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
import gym
import math
import os
import sys

sys.path.append('./neato')
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters


runs_per_net=2
def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(runs_per_net):
        env = gym.make("CartPole-v1")
        env._max_episode_steps = 750
        observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
            action = genome.forward(observation)[0]
            #print(action)
            observation, reward, done, info = env.step(action <= 0.5)
            fitness += reward
        #env.close()
        fitnesses.append(fitness)

    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.max_fitness = 749

    inputs = 4
    outputs = 1
    hidden_layers = 6
    population = 100
    
    brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
    brain.generate()
    
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        current_best = brain.get_fittest()
        true_pop_size = brain.get_population()
        print("Current best genome: {0} | Current best fitness: {1} | Current generation: {2}| Current pop size: {3}".format(
            current_best,
            current_best.get_fitness(), 
            current_gen,
            true_pop_size
        ))

    with open('best_genome', 'wb') as f:
        pickle.dump(current_best, f)

if __name__ == '__main__':
    run()
