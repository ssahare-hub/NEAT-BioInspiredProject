# pip3 install gym
# for gym stuff: 
# apt install xvfb ffmpeg xorg-dev libsdl2-dev swig cmake
# pip3 install gym[box2d]



# To use render environment need to use only one cpu in brain and uncomment render part
#  or it will become frozen. (Probably way to fix this....)
import os
import pickle
import random
import numpy as np
#import cart_pole
import gym
import os
import sys
import matplotlib.pyplot as plt
# import pygame
sys.path.append('./neato')
from neato.genome import Genome
from neato.brain import Brain
from neato.hyperparameters import Hyperparameters, tanh

EPISODE_DURATION = 500
 
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
            # reward = 25* np.exp(-1*(next_observation[0]-1)*(next_observation[0]-1)/0.001)-100*np.abs(10*0.5 - (10*0.5*next_observation[0] + 0.5*0.3333*next_observation[2] * next_observation[2])) + 100*np.abs(10*0.5 - (10*0.5*last_observation[0] + 0.5*0.3333*last_observation[2] * last_observation[2]))
            fitness += reward
            last_observation = next_observation
        fitnesses.append(fitness)
    sys.stdout.flush()
    return np.mean(fitnesses)


def generate_visualized_network(genome: Genome, generation):
    """Generate the positions/colors of the neural network nodes"""
    nodes = {}
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    plt.title(f'Best Genome with fitness {genome.get_fitness()} Network')
    _nodes = genome.get_nodes()
    layer_count = defaultdict(lambda: -1 * genome._inputs)
    index = {}
    for n in _nodes:
        layer_count[_nodes[n].layer] += 1
        index[n] = layer_count[_nodes[n].layer]
    for number in _nodes:
        if genome.is_input(number):
            color = 'blue'
            x = 0.05*NETWORK_WIDTH
            y = HEIGHT/4 + HEIGHT/5 * number
        elif genome.is_output(number):
            color = 'red'
            x = NETWORK_WIDTH-0.05*NETWORK_WIDTH
            y = HEIGHT/2
        else:
            color = 'black'
            x = NETWORK_WIDTH/10 + NETWORK_WIDTH/12 * _nodes[number].layer
            y = HEIGHT/2 + HEIGHT / \
                (layer_count[_nodes[number].layer]+2) * index[number]
        nodes[number] = [(x, y), color]

    print(len(_nodes))
    genes = genome.get_connections()
    sorted_innovations = sorted(genes.keys())
    for innovation in sorted_innovations:
        connection = genes[innovation]
        i, j = connection.in_node.number, connection.out_node.number
        if connection.enabled:
            color = 'green'
        else:
            color = 'red'
        x_values = [nodes[i][0][0], nodes[j][0][0]]
        y_values = [nodes[i][0][1], nodes[j][0][1]]
        ax.plot(x_values, y_values, color=color)

    for n in nodes:
        circle = plt.Circle(nodes[n][0], 5, color=nodes[n][1])
        ax.add_artist(circle)
        # t = txt.Text(nodes[n][0][0] + 10, nodes[n][0][1], str(genome._nodes[n].layer))
        # ax.add_artist(t)
        # t = txt.Text(nodes[n][0][0] - 10, nodes[n][0][1] + 10, str(n), color='red')
        # ax.add_artist(t)
    if not os.path.exists('pendulum_graphs'):
        os.makedirs('pendulum_graphs')
    plt.savefig(f'pendulum_graphs/{generation}._network.png')
    # plt.show()


def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.25
    hyperparams.mutation_probabilities['connection'] = 0.25
    hyperparams.mutation_probabilities['mutate'] = 0.25
    hyperparams.fitness_offset = 0 * EPISODE_DURATION
    hyperparams.max_fitness = hyperparams.fitness_offset
    hyperparams.max_generations = 1200

    inputs = 3
    outputs = 1
    hidden_layers = 9
    population = 500
    if os.path.isfile('pendulum.neat'):
        brain = Brain.load('pendulum')
        brain._hyperparams = hyperparams
    else:    
        brain = Brain(inputs, outputs, hidden_layers, population, hyperparams)
        brain.generate()
        print(hyperparams.max_fitness)

    
    current_best = None
    print("Training...")
    while brain.get_generation() < hyperparams.max_generations:
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        brain.save_fitness_history()
        brain.save_max_fitness_history()
        current_best = brain.get_current_fittest()
        mean_fitness = brain.get_average_fitness()
        print(
            "Mean Fitness: {} | Best Gen Fitness: {} | Species Count: {} |  Current gen: {}".format(
                mean_fitness,
                current_best.get_fitness(), 
                brain.get_species_count(),
                current_gen, 
            )
        )
        sys.stdout.flush()
        print('saving current population')
        brain.save('pendulum')
        generate_visualized_network(current_best, current_gen)
        # NOTE: I wanted to see intermediate results
        # so saving genome whenever it beats the last best
        if current_best.get_fitness() > brain._global_best.get_fitness():
            with open(f'pendulum_best_individual_gen{current_gen}', 'wb') as f:
                pickle.dump(current_best, f)
        brain.update_fittest()
        # break
        plt.figure()
        plt.title('fitness over generations')
        plt.plot(brain.get_fitness_history(),label='average')
        plt.plot(brain.get_max_fitness_history(), label='max')
        plt.savefig(f'pendulum_graphs/progress.png')

    with open('pendulum_best_individual', 'wb') as f:
        pickle.dump(brain._global_best, f)
    

if __name__ == '__main__':
    run()
