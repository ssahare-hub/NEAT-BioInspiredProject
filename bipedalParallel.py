import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
import sys
sys.path.append('./neato')
from neato.genome import Genome
from neato.neato import NeatO, generate_visualized_network
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 500
seed = 0

def evaluate(genome: Genome):
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
            action = genome.forward(observation)
            observation, reward, done, info = env.step(action)
            fitness += reward
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
    hyperparams.mutation_probabilities['weight_perturb'] = 0.8
    hyperparams.mutation_probabilities['weight_set'] = 0.01
    hyperparams.mutation_probabilities['bias_perturb'] = 0.8
    hyperparams.mutation_probabilities['bias_set'] = 0.01
    hyperparams.mutation_probabilities['re-enable'] = 0.01
    hyperparams.fitness_offset = 500
    hyperparams.survival_percentage = 0.2
    hyperparams.max_fitness = hyperparams.fitness_offset + 300
    hyperparams.max_generations = 100

    inputs = 24
    outputs = 4
    hidden_layers = 6
    population = 200
    
    if os.path.isfile('neato_bipedal.neat'):
        neato = NeatO.load('neato_bipedal')
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
            neato.save('neato_bipedal')
        except Exception as e:
            print("Failed to save current neato:")
            print(e)
        try:
            generate_visualized_network(current_best, current_gen, 'bipedal/bipedal_graphs')
            # NOTE: I wanted to see intermediate results
            # so saving genome whenever it beats the last best
            if not os.path.exists('bipedal'):
                os.makedirs('bipedal')
            if not os.path.exists('bipedal/models'):
                os.makedirs('bipedal/models')
            if current_best.get_fitness() > neato._global_best.get_fitness():
                with open(f'bipedal/models/neato_bipedal_gen{current_gen}', 'wb') as f:
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
            plt.savefig(f'bipedal/graphs/progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_network_history(), label='network size')
            plt.legend()
            plt.savefig(f'bipedal/graphs/network_progress.png')
            plt.close()
            plt.figure()
            plt.plot(neato.get_population_history(), label='population size')
            plt.legend()
            plt.savefig(f'bipedal/graphs/population_progress.png')
            plt.close()
        except Exception as e:
            print('progress plots', '='*40)
            print(e)
            print('='*100)


    with open('bipedal/bipedal_cartpole_best_individual', 'wb') as f:
        pickle.dump(neato._global_best, f)

if __name__ == '__main__':
    run()
