from connection import Connection
from connection_history import ConnectionHistory

from node import Node
from genome import Genome
from species import *
from hyperparameters import *

import random
import time

import pygame
from pygame.locals import *

# Constants
WIDTH, HEIGHT = 640, 680
NETWORK_WIDTH = 680

DRAW_NETWORK = True

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)


def generate_visualized_network(genome, nodes):
    """Generate the positions/colors of the neural network nodes"""
    for i in genome.get_nodes():
        if genome.is_input(i):
            color = (0, 0, 255)
            x = 0.05*NETWORK_WIDTH
            y = HEIGHT/4 + HEIGHT/5 * i
        elif genome.is_output(i):
            color = (255, 0, 0)
            x = NETWORK_WIDTH-0.05*NETWORK_WIDTH
            y = HEIGHT/2
        else:
            color = (0, 0, 0)
            # x = random.randint(NETWORK_WIDTH/3, int(NETWORK_WIDTH * (4.0/5)))
            x = NETWORK_WIDTH/10 + NETWORK_WIDTH/12 * genome._nodes[i].layer
            y = random.randint(20, HEIGHT-20)
        nodes[i] = [(int(x), int(y)), color]

def render_visualized_network(genome, nodes, display):
    """Render the visualized neural network"""
    genes = genome.get_connections()
    for innovation in genes:
        connection = genes[innovation]
        if connection.enabled: # Enabled or disabled connection
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        i, j = connection.in_node.number, connection.out_node.number
        pygame.draw.line(display, color, nodes[i][0], nodes[j][0], 3)

    for n in nodes:
        pygame.draw.circle(display, nodes[n][1], nodes[n][0], 7)


def main():
    pygame.init()
    display = pygame.display.set_mode((NETWORK_WIDTH, HEIGHT), 0, 32)
    network_display = pygame.Surface((NETWORK_WIDTH, HEIGHT))
    
    hyperparameters = Hyperparameters()
    
    default_activation = hyperparameters.default_activation
    ch = ConnectionHistory(4,1,9)
    g = Genome(ch,default_activation,True)
    


    
    # print(len(g._nodes))
    while True:
        network_display.fill(WHITE)
        nodes = {}
        generate_visualized_network(g, nodes)
        if DRAW_NETWORK:
            render_visualized_network(g, nodes, network_display)
        display.blit(network_display, (0, 0))
        pygame.display.update()
        try:
            g.mutate(hyperparameters.mutation_probabilities)
        except Exception as e:
            print(e)
            pygame.image.save(display,'graph.jpg')
            break
        time.sleep(0.5)
        output = g.forward([10,1,2,3])
        print(output)
        print(">==============================================================================================<")
        for n in (g._connections):
            g._connections[n].showConn()

        # genes = g.get_connections()
        # for connection in genes:
        #     print(connection)
        # for i in g.get_nodes():
        #     print(g._nodes[i].number,g._nodes[i].layer)

        # nodes = {}
        # generate_visualized_network(g, nodes)

        # if DRAW_NETWORK:
        #     render_visualized_network(g, nodes, network_display)
        # display.blit(network_display, (0, 0))
        # pygame.display.update()
        # time.sleep(2)

        
        # print(g.connections[0].in_node.number)




if __name__ == "__main__":
    main()