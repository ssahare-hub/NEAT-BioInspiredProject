# %%
from collections import defaultdict
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

def generate_visualized_network(genome: Genome, generation):
    """Generate the positions/colors of the neural network nodes"""
    nodes = {}
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    plt.title(f'Best Genome with fitness {genome.get_fitness()} Network')
    _nodes = genome.get_nodes()
    layer_count = defaultdict(lambda: -1 * genome._inputs )
    index = {}
    max_nodes_per_layer = genome._inputs
    for n in _nodes:
        layer_count[_nodes[n].layer] += 1
        if layer_count[_nodes[n].layer] == 0:
            layer_count[_nodes[n].layer] += 1
        index[n] = layer_count[_nodes[n].layer]
        max_nodes_per_layer = max( max_nodes_per_layer, layer_count[_nodes[n].layer] + genome._inputs )
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
            t = max( (layer_count[_nodes[number].layer]) * 2.5, max_nodes_per_layer)
            y = HEIGHT/4 + (HEIGHT / t) * index[number]
        nodes[number] = [(x, y), color]

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
        # t = txt.Text(nodes[n][0][0] + 10, nodes[n][0][1], str(index[n]))
        # ax.add_artist(t)
        # t = txt.Text(nodes[n][0][0] - 10, nodes[n][0][1] - 10, str(genome._nodes[n].layer), color='red')
        # ax.add_artist(t)
    if not os.path.exists('pendulum_graphs'):
        os.makedirs('pendulum_graphs')
    plt.show()
    # plt.savefig(f'pendulum_graphs/{generation}._network.png')
    # plt.clf()


with open('mountaincar_best_individual', 'rb') as f:
    genome = pickle.load(f)
    generate_visualized_network(genome, 1)

# %%
