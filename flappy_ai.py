# A Flappy Bird clone that learns to play Flappy Bird
# This is a simple demonstration of the capabilities of the NEAT algorithm
# Requires Pygame to run
import sys
import os
import random
from neato.genome import Genome
import pygame
from pygame.locals import *

sys.path.append('./neato')
from neato.neato import NeatO
from neato.hyperparameters import Hyperparameters

 
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
GREEN = (0,255,0)
BLUE = (0, 0, 200)


class Bird:
    def __init__(self, x, y):
        self.dim = [20, 20]
        self.pos = [x, y]
        self.vel = [0, 0]

        self.bounce = 8
        self.gravity = 0.5
        self.terminal_vel = 10
        self.onGround = False
        self.alive = True
        self.jump = False

    def draw(self, surf):
        center_pos = (int(self.pos[0]-self.dim[0]//2), int(self.pos[1]-self.dim[1]//2))
        pygame.draw.rect(surf, WHITE, (center_pos, self.dim))

    def update(self):
        if self.vel[1] < self.terminal_vel:
            self.vel[1] += self.gravity

        self.pos[1] += self.vel[1]

        if self.jump:
            self.vel[1] = -self.bounce

        self.jump = False
    
    def colliding(self, pipe):
        x = abs(self.pos[0]-pipe.pos[0]) < (self.dim[0]+pipe.dim[0])/2.0
        y = abs(self.pos[1]-pipe.pos[1]) < (self.dim[1]+pipe.dim[1])/2.0
        if x and y:
            return True
        return False

    def kill(self):
        self.alive = False


class Pipe:
    def __init__(self, x, y):
        self.dim = [50, 480]
        self.pos = [x, y]
        self.vel = [-2, 0]

        self.alive = True

    def draw(self, surf):
        center_pos = (int(self.pos[0]-self.dim[0]//2), int(self.pos[1]-self.dim[1]//2))
        pygame.draw.rect(surf, YELLOW, (center_pos, self.dim))

    def update(self):
        self.pos[0] += self.vel[0]
        if self.pos[0] < -self.dim[0]:
            self.kill()

    def kill(self):
        self.alive = False


def generate_pipes(pipes):
    """Generate a column of pipes for the bird to fly through"""
    y = random.randint(150, HEIGHT-150)
    th = 150

    p1 = Pipe(WIDTH, y-(th/2+HEIGHT/2))
    p2 = Pipe(WIDTH, y+(th/2+HEIGHT/2))
    pipes.extend([p1, p2])

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
            x = NETWORK_WIDTH/10 + NETWORK_WIDTH/(genome._hidden_layers+3) * genome._nodes[i].layer
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

    display = pygame.display.set_mode((WIDTH+NETWORK_WIDTH, HEIGHT), 0, 32)
    network_display = pygame.Surface((NETWORK_WIDTH, HEIGHT))

    clock = pygame.time.Clock()
    timer = 0

    inputs = 4
    outputs = 1
    hidden_layers = 6
    population = 100

    player = Bird(WIDTH/4, HEIGHT/2)
    pipes = []
    ahead = []

    # Load the bird's brain
    if os.path.isfile('flappy_bird.neat'):
        brain = NeatO.load('flappy_bird')
    else:
        hyperparams = Hyperparameters()
        hyperparams.delta_threshold = 1.5

        hyperparams.mutation_probabilities['node'] = 0.05
        hyperparams.mutation_probabilities['connection'] = 0.05

        brain = NeatO(inputs, outputs, hidden_layers, population, hyperparams)
        brain.initialize()

    inputs = [0, 0, 0, 0]
    output = 0
    nodes = {}
    generate_visualized_network(brain.get_current(), nodes)

    while True:
        display.fill(BLUE)
        network_display.fill(WHITE)

        # Simulation logic
        if player.alive:
            player.draw(display)
            player.update()

            timer += 1
            if timer%150 == 0:
                generate_pipes(pipes)

            for p in pipes:
                p.draw(display)
                if p.alive:
                    p.update()
                    if player.colliding(p):
                        player.kill()
                else:
                    pipes.remove(p)

            # Player height kill-barrier
            if player.pos[1] >= HEIGHT-player.dim[1]/2 or player.pos[1] <= 0:
                player.kill()
        else:
            timer = 0
            pipes = []
            player = Bird(WIDTH/4, HEIGHT/2)

            if AI:
                # Save the bird's brain
                brain.save('flappy_bird')
                brain.next_iteration()
                nodes = {}
                generate_visualized_network(brain.get_current(), nodes)

        # Train the neural network
        if AI:
            ahead = [p for p in pipes if p.pos[0]-player.pos[0] >= 0][:2]
            if len(ahead) > 0:
                # inputs = [x-dist, player height, top-pipe, bottom-pipe]
                inputs = [(ahead[0].pos[0]-player.pos[0])/WIDTH, 
                          (player.pos[1])/HEIGHT,
                          (ahead[0].pos[1]+ahead[0].dim[1]/2)/HEIGHT,
                          (ahead[1].pos[1]-ahead[1].dim[1]/2)/HEIGHT]
            else:
                inputs = [0, player.pos[1]/HEIGHT, 0, 0]

            if brain.should_evolve():
                genome = brain.get_current()
                output = genome.forward(inputs)[0]
                # print(output)
                genome.set_fitness(timer)
                player.jump = (output <= 0.5)

                if DRAW_NETWORK:
                    render_visualized_network(genome, nodes, network_display)

        # Handle user events events
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit()
            if e.type == KEYDOWN:
                if e.key == K_SPACE and not AI:
                    player.jump = True

        # Update display and its caption
        display.blit(network_display, (WIDTH, 0))
        pygame.display.set_caption("GNRTN : {0}; SPC : {1}; CRRNT : {2}; FIT : {3}; MXFIT : {4}; OUT : {5} IN : {6}".format(
                                            brain.get_generation()+1, 
                                            brain.get_current_species()+1, 
                                            brain.get_current_genome()+1,
                                            timer,
                                            brain.get_all_time_fittest().get_fitness(),
                                            round(output, 3),
                                            [round(i, 3) for i in inputs]
        ))
        pygame.display.update()

        # Uncomment this to cap the framerate
        # clock.tick(1000)

if __name__ == "__main__":
    main()