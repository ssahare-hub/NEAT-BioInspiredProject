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

def generate_visualized_network(genome: Genome,generation):
    """Generate the positions/colors of the neural network nodes"""
    nodes = {}
    # for i in genome.get_nodes():
    genes = genome.get_connections()
    fig, ax = plt.subplots()
    plt.title(f'Best Genome with fitness {genome.get_fitness()} Network')
    for innovation in genes:
        connection = genes[innovation]
        i, j = connection.in_node.number, connection.out_node.number
        connection_nodes = [i, j]
        for number in connection_nodes:
            if genome.is_input(number):
                color = 'blue'
                x = 0.05*NETWORK_WIDTH
                y = HEIGHT/4 + HEIGHT/5 * i
            elif genome.is_output(number):
                color = 'red'
                x = NETWORK_WIDTH-0.05*NETWORK_WIDTH
                y = HEIGHT/2
            else:
                color = 'black'
                # x = random.randint(NETWORK_WIDTH/3, int(NETWORK_WIDTH * (4.0/5)))
                x = NETWORK_WIDTH/10 + NETWORK_WIDTH/(genome._hidden_layers+3) * genome._nodes[i].layer
                y = random.randint(20, HEIGHT-20)
            nodes[number] = [(int(x), int(y)), color]
        if connection.enabled: # Enabled or disabled connection
            color = 'green'
        else:
            color = 'red'
        x_values = [nodes[i][0][0], nodes[j][0][0]]
        y_values = [nodes[i][0][1], nodes[j][0][1]]
        ax.plot(x_values, y_values, color = color)
    for n in nodes:
        circle = plt.Circle(nodes[n][0], NETWORK_WIDTH/(genome._hidden_layers+100), color=nodes[n][1])
        ax.add_patch(circle)
    if not os.path.exists('pendulum_graphs'):
        os.makedirs('pendulum_graphs')
    plt.savefig(f'pendulum_graphs/{generation}._network.png')


def run():

    hyperparams = Hyperparameters()
    #hyperparams.max_generations = 300
    hyperparams.default_activation = tanh
    hyperparams.delta_threshold = 0.75
    hyperparams.mutation_probabilities['node'] = 0.05
    hyperparams.mutation_probabilities['connection'] = 0.05
    hyperparams.mutation_probabilities['mutate'] = 0.05
    hyperparams.fitness_offset = 0 * EPISODE_DURATION
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
    while brain.get_generation() < hyperparams.max_generations:
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        brain.update_fittest()
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
        # break

    with open('pendulum_best_individual', 'wb') as f:
        pickle.dump(current_best, f)
    
    plt.figure()
    plt.title('fitness over generations')
    plt.plot(brain.get_fitness_history(),label='average')
    plt.plot(brain.get_max_fitness_history(), label='max')
    plt.show()

if __name__ == '__main__':
    run()
