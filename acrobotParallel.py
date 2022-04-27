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

def evaluate(genome:Genome):
    """Evaluates the current genome."""
    try:
        fitnesses = []
        for i in range(5):
            env = gym.make("Acrobot-v1")
            last_observation = env.reset()
            fitness = 0.
            done = False
            while not done:
                output = genome.forward(last_observation)[0]
                if output <= 0.3:
                    action = -1
                elif 0.3 < output <= 0.6:
                    action = 0
                else:
                    action = 1
                next_observation, reward, done, info = env.step(action)
                fitness += reward
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
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 2
    hyperparams.mutation_probabilities['node'] = 0.2
    hyperparams.mutation_probabilities['connection'] = 0.2
    hyperparams.mutation_probabilities['mutate'] = 0.3
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.7
    hyperparams.mutation_probabilities['bias_set'] = 0.00
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.fitness_offset = 1000
    hyperparams.max_fitness = 900
    hyperparams.max_generations = 200
    hyperparams.survival_percentage = 0.25

    inputs = 6
    outputs = 1
    hidden_layers = 9
    population = 100
    if os.path.isfile('neato_acrobot.neat'):
        neato = NeatO.load('neato_acrobot')
        neato._hyperparams = hyperparams
    else:
        neato = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        neato.initialize()
        print(hyperparams.max_fitness)

    
    current_best = None
    print("Training...")
    while neato.should_evolve():
        try:
            neato.evaluate_parallel(evaluate)

            # Print training progress
            current_gen = neato.get_generation()
            current_best = neato.get_current_fittest()
            mean_fitness = neato.get_fitness_history()[-1]
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
            neato.save('neato_acrobot')
        except Exception as e:
            print("Failed to save current neato:")
            print(e)
        try:
            generate_visualized_network(current_best, current_gen, 'acrobot/acrobot_graphs/')
            if not os.path.exists('acrobot'):
                os.makedirs('acrobot')
            if not os.path.exists('acrobot/models'):
                os.makedirs('acrobot/models')
            with open(f'acrobot/models/neato_acrobot_{current_gen}_{abs(round(current_best.get_fitness()))}', 'wb') as f:
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
            plt.savefig(f'acrobot/graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'acrobot/graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'acrobot/graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)
        # break

    with open('acrobot/neato_acrobot_best_individual', 'wb') as f:
        pickle.dump(neato._global_best, f)
    

if __name__ == '__main__':
    run()
