import pickle
import gym
import math
import sys

sys.path.append('./neato')
from neato.genome import Genome


def run():
    gen = 35
    with open('mountaincar_best_individual', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()

    env = gym.make("MountainCarContinuous-v0") # BipedalWalker-v3
    for _ in range(5):
        fitness = 0.0
        done = False
        last_observation = env.reset()
        while not done:
            action = genome.forward(last_observation)
            next_observation, reward, done, info = env.step(action)
            reward = reward = 100*((math.sin(3*next_observation[0]) * 0.0025 + 0.5 * next_observation[1] * next_observation[1]) - (math.sin(3*last_observation[0]) * 0.0025 + 0.5 * last_observation[1] * last_observation[1]))
            last_observation = next_observation
            fitness += reward
            env.render()

        print(fitness)

if __name__ == '__main__':
    run()