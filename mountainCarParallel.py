import os
import pickle
import math
import numpy as np
import gym
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('./neato')
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 500

def evaluate(genome):
    """Evaluates the current genome."""
    fitnesses = []
    for _ in range(5):
        env = gym.make("MountainCarContinuous-v0")
        # env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()

        fitness = 0.
        done = False
        while not done:
            #if i == 0:
            #    env.render()
            action = genome.forward(last_observation)

            next_observation, reward, done, info = env.step(action)
            reward = reward = 100*((math.sin(3*next_observation[0]) * 0.0025 + 0.5 * next_observation[1] * next_observation[1]) - (math.sin(3*last_observation[0]) * 0.0025 + 0.5 * last_observation[1] * last_observation[1]))
            fitness += reward
            last_observation = next_observation
            # fitness += 1 if abs(observation[1]) >= 0.05 else 0
            # add a positive reward for being alive
        fitnesses.append(fitness)
    return np.mean(fitnesses)

def run():

    hyperparams = Hyperparameters()
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.25
    hyperparams.fitness_offset = 10
    hyperparams.max_fitness = hyperparams.fitness_offset
    hyperparams.max_generations = 300

    inputs = 2
    outputs = 1
    hidden_layers = 6
    population = 400
    if os.path.isfile('mountaincar.neat'):
        brain = Brain.load('mountaincar')
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
        current_best = brain.get_fittest()
        mean_fitness = brain.get_average_fitness()
        brain.save_fitness_history(mean_fitness)
        print(
            "Mean Fitness: {4} | Current Accuracy: {0} | Current species: {1} | Current genome: {2} | Current gen: {3}".format(
                current_best.get_fitness(), 
                brain.get_current_species()+1, 
                brain.get_current_genome()+1,
                current_gen, 
                mean_fitness
            )
        )
        sys.stdout.flush()
        print('saving current population')
        brain.save('mountaincar')
    print('done')
    with open('mountaincar_best_individual', 'wb') as f:
        pickle.dump(current_best, f)
    
    plt.title('fitness over generations')
    plt.plot(brain.get_fitness_history())
    plt.show()

if __name__ == '__main__':
    run()
