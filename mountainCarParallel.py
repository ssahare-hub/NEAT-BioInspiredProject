import os
import pickle
import math
import numpy as np
import gym
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('./neato')
from neato.genome import Genome
from neato.neato import NeatO, generate_visualized_network
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 500

def evaluate(genome: Genome):
    """Evaluates the current genome."""
    fitnesses = []
    for _ in range(5):
        env = gym.make("MountainCarContinuous-v0")
        env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()
        fitness = 0.
        done = False
        while not done:
            action = genome.forward(last_observation)

            next_observation, reward, done, info = env.step(action)
            reward = reward = 100*((math.sin(3*next_observation[0]) * 0.0025 + 0.5 * next_observation[1] * next_observation[1]) - (math.sin(3*last_observation[0]) * 0.0025 + 0.5 * last_observation[1] * last_observation[1]))
            fitness += reward
            last_observation = next_observation
        fitnesses.append(fitness)
    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 3.85
    hyperparams.mutation_probabilities['node'] = 0.2
    hyperparams.mutation_probabilities['connection'] = 0.2
    hyperparams.mutation_probabilities['mutate'] = 0.75
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.1
    hyperparams.mutation_probabilities['bias_perturb'] = 0.8
    hyperparams.mutation_probabilities['bias_set'] = 0.1
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.perturbation_range['weight_perturb_max'] = 2
    hyperparams.perturbation_range['weight_perturb_min'] = -2
    hyperparams.survival_percentage = 0.2


    hyperparams.fitness_offset = 0
    hyperparams.max_fitness = hyperparams.fitness_offset+0.703
    hyperparams.max_generations = 300
    hyperparams.distance_weights['matching_connections'] = 0.6

    inputs = 2
    outputs = 1
    hidden_layers = 6
    population = 1000
    if os.path.isfile('mountain_car.neat'):
        neato = NeatO.load('mountain_car')
        neato._hyperparams = hyperparams
    else:    
        neato = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        neato.initialize()
        print(hyperparams.max_fitness)

    current_best = None
    print("Training...")
    while neato.should_evolve():
        neato.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = neato.get_generation()
        current_best = neato.get_current_fittest()
        
        mean_fitness = neato.get_fitness_history()[-1]
        print(
            "Mean Fitness: {} | Current Accuracy: {} | Species Count: {} | Population count: {} | Current gen: {}".format(
                mean_fitness,
                current_best.get_fitness(), 
                neato.get_species_count(), 
                neato.get_population(),
                current_gen
            )
        )
        sys.stdout.flush()
        print('saving current population')
        neato.save('mountain_car')
        generate_visualized_network(current_best, current_gen,'mountain_car/graphs/')
        # NOTE: I wanted to see intermediate results
        # so saving genome whenever it beats the last best
        if not os.path.exists('mountain_car/models'):
            os.makedirs('mountain_car/models')
        if current_best.get_fitness() >= neato._global_best.get_fitness():
            with open(f'mountain_car/models/mountain_car_gen{current_gen}', 'wb') as f:
                pickle.dump(current_best, f)
        neato.update_fittest()
        # break
        try:
            plt.figure()
            plt.title('fitness over generations')
            plt.plot(neato.get_fitness_history(),label='average')
            plt.plot(neato.get_max_fitness_history(), label='max')
            plt.axhline(y=hyperparams.max_fitness, color='red', linestyle='-', label='desired max fitness')
            plt.legend()
            plt.savefig(f'mountain_car/graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'mountain_car/graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'mountain_car/graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)

    print('done')
    with open('mountain_car_best_individual', 'wb') as f:
        pickle.dump(neato.get_all_time_fittest(), f)
    

if __name__ == '__main__':
    run()
