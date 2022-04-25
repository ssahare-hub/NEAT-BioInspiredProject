# %%
import os
import random
from neato.brain import Brain
import matplotlib.pyplot as plt
import matplotlib.text as txt
import pickle

from neato.genome import Genome
# Constants
WIDTH, HEIGHT = 640, 480
NETWORK_WIDTH = 480

# Flags
AI = True
DRAW_NETWORK = True

# %%
def generate_visualized_network(genome: Genome,generation):
    """Generate the positions/colors of the neural network nodes"""
    nodes = {}
    genes = genome.get_connections()
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca()
    ax.axis('off')
    plt.title(f'Best Genome with fitness {genome.get_fitness()} Network')
    for innovation in genes:
        connection = genes[innovation]
        i, j = connection.in_node.number, connection.out_node.number
        connection_nodes = [i, j]
        print(connection_nodes)
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
                x = NETWORK_WIDTH/10 + NETWORK_WIDTH/(genome._hidden_layers+3) * genome._nodes[i].layer
                y = random.randint(20, HEIGHT-20)
            nodes[number] = [(x, y), color]

    for innovation in genes:
        connection = genes[innovation]
        i, j = connection.in_node.number, connection.out_node.number
        if connection.enabled: 
            color = 'green'
        else:
            color = 'red'
        x_values = [nodes[i][0][0], nodes[j][0][0]]
        y_values = [nodes[i][0][1], nodes[j][0][1]]
        ax.plot(x_values, y_values, color = color)
    for n in nodes:
        circle = plt.Circle(nodes[n][0], NETWORK_WIDTH/(genome._hidden_layers+80), color=nodes[n][1])
        ax.add_artist(circle)
    if not os.path.exists('pendulum_graphs'):
        os.makedirs('pendulum_graphs')
    plt.savefig(f'pendulum_graphs/{generation}._network.png')


with open('pendulum204_best_individual', 'rb') as f:
        genome = pickle.load(f)
        generate_visualized_network(genome, 1)

# %%
