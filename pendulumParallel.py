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
import matplotlib.pyplot as plt

sys.path.append('./neato')
from neato.genome import Genome
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 500


def evaluate(genome:Genome):
    """Evaluates the current genome."""
    fitnesses = []
    for i in range(5):
        env = gym.make("Pendulum-v1")
        env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
            action = genome.forward(last_observation)[0]

            next_observation, reward, done, info = env.step([action*2])
            #print('observation: ', observation)
            #print("action ",action)
            #print("reward ", reward)
            # reward = 25* np.exp(-1*(next_observation[0]-1)*(next_observation[0]-1)/0.001)-100*np.abs(10*0.5 - (10*0.5*next_observation[0] + 0.5*0.3333*next_observation[2] * next_observation[2])) + 100*np.abs(10*0.5 - (10*0.5*last_observation[0] + 0.5*0.3333*last_observation[2] * last_observation[2]))
            # print(reward)
            fitness += reward
            last_observation = next_observation
            # add a positive reward for being alive
        #env.close()
        # print("fitness ", fitness)
        fitnesses.append(fitness)
    # print("mean fitness ", np.mean(fitnesses))
    # print("global innovation rn is", genome.connection_history.global_innovation_count)
    sys.stdout.flush()
    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.05
    hyperparams.fitness_offset = 17 * EPISODE_DURATION
    hyperparams.max_fitness = hyperparams.fitness_offset
    hyperparams.max_generations = 600

    inputs = 3
    outputs = 1
    hidden_layers = 9
    population = 1000
    if os.path.isfile('pendulum.neat'):
        brain = Brain.load('pendulum')
        brain._hyperparams = hyperparams
    else:    
        brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
        brain.generate()
        print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
    fitness_history = []
    while brain.get_generation() < hyperparams.max_generations:
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        brain.update_fittest()
        brain.save_fitness_history()
        current_best = brain.get_fittest()
        mean_fitness = brain.get_average_fitness()
        print(
            "Mean Fitness: {4} | Best Gen Accuracy: {0} | Species Count: {1} | Current genome: {2} | Current gen: {3}".format(
                brain.get_maximum_fitness(), 
                brain.get_species_count(), 
                brain.get_current_genome()+1,
                current_gen, 
                mean_fitness
            )
        )
        sys.stdout.flush()
        print('saving current population')
        brain.save('pendulum')

    with open('pendulum_best_individual', 'wb') as f:
        pickle.dump(current_best, f)
    
    plt.title('fitness over generations')
    plt.plot(brain.get_fitness_history())
    plt.show()

if __name__ == '__main__':
    run()
