import multiprocessing
import os
import pickle

import neat
import numpy as np
#import cart_pole
import gym
import visualize

runs_per_net = 4
# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
        env = gym.make("Acrobot-v1")
        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            output = net.activate(observation)[0]
            if output <= 0.3:
                action = -1
            elif 0.3 < output <= 0.6:
                action = 0
            else:
                action = 1
            observation, reward, done, info = env.step(action)
            fitness += reward
        fitness += 1000

        fitnesses.append(fitness)

    return np.mean(fitnesses)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'acrobotNeatPythonconfig')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    if not os.path.exists('acrobotNP'):
        os.makedirs('acrobotNP')
    # Save the winner.
    with open('acrobotNP/acrobotNeatPythonBest', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

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




if __name__ == '__main__':
    run()