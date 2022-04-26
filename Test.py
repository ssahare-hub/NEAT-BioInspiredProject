
import random
import time

import pygame
from pygame.locals import *

import sys

sys.path.append('./neato')
from neato.genome import *
from neato.species import *
from neato.hyperparameters import *
from neato.connection_history import *

# Constants
WIDTH, HEIGHT = 640, 680
NETWORK_WIDTH = 380

DRAW_NETWORK = True

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)


def generate_visualized_network(genome: Genome, nodes):
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

def render_visualized_network(genome: Genome, nodes, display):
    """Render the visualized neural network"""
    genes = genome.get_connections()
    sorted_innovations = sorted(genes,key=lambda g: g)
    for innovation in sorted_innovations:
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
    display = pygame.display.set_mode((3*NETWORK_WIDTH, HEIGHT), 0, 32)
    p1_network_display = pygame.Surface((NETWORK_WIDTH, HEIGHT))
    p2_network_display = pygame.Surface((2*NETWORK_WIDTH, HEIGHT))
    c1_network_display = pygame.Surface((3*NETWORK_WIDTH, HEIGHT))
    
    hyperparameters = Hyperparameters()
    
    default_activation = hyperparameters.default_activation
    ch = ConnectionHistory(4,1,9)
    g = Genome(ch,default_activation,True)
    g2 = Genome(ch,default_activation,True)
    

    for i in range(100):
        p1_network_display.fill(WHITE)
        p2_network_display.fill(WHITE)
        c1_network_display.fill(WHITE)
        p1_nodes = {}
        p2_nodes = {}
        
        generate_visualized_network(g, p1_nodes)
        generate_visualized_network(g2, p2_nodes)
        if DRAW_NETWORK:
            render_visualized_network(g, p1_nodes, p1_network_display)
            render_visualized_network(g2, p2_nodes, p2_network_display)
        display.blit(p1_network_display, (0, 0))
        display.blit(p2_network_display, (NETWORK_WIDTH, 0))
        
        pygame.display.update()
        try:
            g.mutate(hyperparameters)
            g2.mutate(hyperparameters)

            g._fitness = g.forward([10,1,2,3])
            g2._fitness = g2.forward([10,1,2,3])
            if i>2 and not i%10:
                print("crossover",i)
                c1 = genomic_crossover(g,g2)
                for n in sorted(c1._connections,key=lambda g: g):
                    c1._connections[n].showConn()
                for n in c1._nodes:
                    print(n,c1._nodes[n].number)
                
        except Exception as e:
            print(e)
            pygame.image.save(display,'graph.jpg')
            break
        
        if 'c1' in locals():
            c1_nodes = {}
            generate_visualized_network(c1, c1_nodes)
            render_visualized_network(c1, c1_nodes, c1_network_display)
            display.blit(c1_network_display, (2*NETWORK_WIDTH, 0))
            pygame.display.update()
            print(g.forward([10,1,2,3]),g2.forward([10,1,2,3]),c1.forward([10,1,2,3]))
        time.sleep(0.5)
        print(">==============================================================================================<")
        # for n in (g._connections):
        #     g._connections[n].showConn()

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