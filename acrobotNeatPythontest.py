import os
import pickle
import neat
import gym 
import numpy as np
import visualize

# load the winner
with open('acrobotNP/acrobotNeatPythonBest', 'rb') as f:
    winner = pickle.load(f)

print('Loaded genome:')
print(winner)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'acrobotNeatPythonconfig')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(winner, config)


env = gym.make("Acrobot-v1")
observation = env.reset()

pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
visualize.plot_stats(stats, ylog=True, view=False, filename="acrobotNP/feedforward-fitness.svg")
visualize.plot_species(stats, view=False, filename="acrobotNP/feedforward-speciation.svg")

node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
visualize.draw_net(config, winner, False, node_names=node_names)

visualize.draw_net(config, winner, view=False, node_names=node_names,
                    filename="acrobotNP/winner-feedforward.gv")
visualize.draw_net(config, winner, view=False, node_names=node_names,
                    filename="acrobotNP/winner-feedforward-enabled.gv", show_disabled=False)
visualize.draw_net(config, winner, view=False, node_names=node_names,
                    filename="acrobotNP/winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


done = False
while not done:
    # next_observation, reward, done, info = env.step(action)
    output = net.activate(observation)[0]
    if output <= 0.3:
        action = -1
    elif 0.3 < output <= 0.6:
        action = 0
    else:
        action = 1
    observation, reward, done, info = env.step(action)
    env.render()