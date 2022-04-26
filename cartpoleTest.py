import pickle
import gym
import sys
sys.path.append('./neato')
from neato.genome import Genome


def run():
    with open('cartpole/neato_cartpole_best_individual', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()

    env = gym.make("CartPole-v1") # BipedalWalker-v3
    observation = env.reset()

    done = False
    while not done:
        action = genome.forward(observation)[0]

        observation, reward, done, info = env.step(action <= 0.5)
        env.render()

if __name__ == '__main__':
    run()