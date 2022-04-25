import multiprocessing
import os
import pickle

import matplotlib.pyplot as plt
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
            #print(reward)
        #env.close()
        #print("fitness ", fitness)
        fitnesses.append(fitness+100)
    sys.stdout.flush()
    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    hyperparams.max_generations = 50
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 1.5
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.05
    #hyperparams.fitness_offset = 100 #*EPISODE_DURATION
    #hyperparams.max_fitness = hyperparams.fitness_offset

    inputs = 24
    outputs = 4
    hidden_layers = 6
    population = 200
    
    if os.path.isfile('bipedal.neat'):
        brain = Brain.load('bipedal')
        brain._hyperparams = hyperparams
    else:    
        brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
        brain.generate()
        print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        
        current_best = brain.get_current_fittest()
        mean_fitness = brain.get_average_fitness()
        print(
            "Mean Fitness: {} | Current Accuracy: {} | Species Count: {} | Population count: {} | Current gen: {}".format(
                mean_fitness,
                current_best.get_fitness(), 
                brain.get_species_count(), 
                brain.get_population(),
                current_gen
            )
        )
        sys.stdout.flush()
        #print('saving current population')
        brain.save('bipedal')
        if current_best.get_fitness() > brain._global_best.get_fitness():
            with open(f'Bipedal_best_individual_gen{current_gen}', 'wb') as f:
                pickle.dump(current_best, f)

        brain.update_fittest()

    print('done')
    with open('bipedal_best_individual', 'wb') as f:
        pickle.dump(brain.get_all_time_fittest, f)

    plt.title('fitness over generations')
    plt.plot(brain.get_fitness_history())
    plt.show()

if __name__ == '__main__':
    run()
