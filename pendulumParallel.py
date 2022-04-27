# To use render environment need to use only one cpu in neato and uncomment render part
#  or it will become frozen. (Probably way to fix this....)
from collections import defaultdict
import os
import pickle
import random
import numpy as np
#import cart_pole
from collections import defaultdict
import gym
import os
import sys
import matplotlib.pyplot as plt
# import pygame
sys.path.append('./neato')
from neato.genome import Genome
from neato.neato import NeatO, generate_visualized_network
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 200
 
# Constants
WIDTH, HEIGHT = 640, 480
NETWORK_WIDTH = 480

# Flags
AI = True
DRAW_NETWORK = True

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

def evaluate(genome:Genome):
    """Evaluates the current genome."""
    try:
        fitnesses = []
        for i in range(10):
            env = gym.make("Pendulum-v1")
            env._max_episode_steps = EPISODE_DURATION
            last_observation = env.reset()
            fitness = 0.
            done = False
            while not done:
                action = genome.forward(last_observation)[0]
                next_observation, reward, done, info = env.step([action * 2])
                # reward = np.exp(-1*(next_observation[0]-1)*(next_observation[0]-1)/0.001)-100*np.abs(10*0.5 - (10*0.5*next_observation[0] + 0.5*0.3333*next_observation[2] * next_observation[2])) + 100*np.abs(10*0.5 - (10*0.5*last_observation[0] + 0.5*0.3333*last_observation[2] * last_observation[2]))
                # fitness += reward
                encourage_vertical = (next_observation[0] + 1) ** 2
                discourage_horizontal = (abs(next_observation[1]) - 1) ** 2
                discourage_speed = action ** 2
                fitness += (reward + encourage_vertical - discourage_horizontal - discourage_speed)
                last_observation = next_observation
            fitnesses.append(fitness)
        sys.stdout.flush()
        return np.mean(fitnesses)
    except Exception as e:
        print('Evaluate','='*100)
        print(e)
        return 0

def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 3
    hyperparams.mutation_probabilities['node'] = 0.2
    hyperparams.mutation_probabilities['connection'] = 0.2
    hyperparams.mutation_probabilities['mutate'] = 0.3
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.7
    hyperparams.mutation_probabilities['bias_set'] = 0.00
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.fitness_offset = 1000
    hyperparams.max_fitness = 2000
    hyperparams.max_generations = 100
    hyperparams.survival_percentage = 0.25

    inputs = 3
    outputs = 1
    hidden_layers = 12
    population = 1000
    if os.path.isfile('neato_pendulum.neat'):
        neato = NeatO.load('neato_pendulum')
        neato._hyperparams = hyperparams
    else:
        neato = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        neato.initialize()
        print(hyperparams.max_fitness)

    
    current_best = None
    print("Training...")
    # while neato.get_generation() < hyperparams.max_generations:
    while neato.should_evolve():
        try:
            neato.evaluate_parallel(evaluate)
            # Print training progress
            current_gen = neato.get_generation()
            
            mean_fitness = neato.get_fitness_history()[-1]
            current_best = neato.get_current_fittest()
            print(
                "Mean Fitness: {} | Best Gen Fitness: {} | Species Count: {} |  Current gen: {}".format(
                    mean_fitness,
                    current_best.get_fitness(), 
                    neato.get_species_count(),
                    current_gen, 
                )
            )
            sys.stdout.flush()
            print('saving current population')
        except Exception as e:
            print('pre-saving', '-'*100)
            print(e)
        try:
            neato.save('neato_pendulum')
        except Exception as e:
            print("Failed to save current neato:")
            print(e)
        try:
            if not os.path.exists('pendulum'):
                os.makedirs('pendulum')
            if not os.path.exists('pendulum/models'):
                os.makedirs('pendulum/models')
            generate_visualized_network(current_best, current_gen, 'pendulum/pendulum_graphs/')
            with open(f'pendulum/models/neato_pendulum_{current_gen}_{abs(round(current_best.get_fitness()))}', 'wb') as f:
                    pickle.dump(current_best, f)
            neato.update_fittest()
        except Exception as e:
            print('Network', '='*40)
            print(e)
            print('='*100)
        # break
        try:
            plt.figure()
            plt.title('fitness over generations')
            plt.plot(neato.get_fitness_history(),label='average')
            plt.plot(neato.get_max_fitness_history(), label='max')
            plt.axhline(y=hyperparams.max_fitness, color='red', linestyle='-', label='desired max fitness')
            plt.legend()
            plt.savefig(f'pendulum/graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'pendulum/graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'pendulum/graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)
        # break

    with open('pendulum/neato_pendulum_best_individual', 'wb') as f:
        pickle.dump(neato._global_best, f)
    

if __name__ == '__main__':
    run()
