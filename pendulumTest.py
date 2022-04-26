import pickle
import gym
import sys
sys.path.append('./neato')
from neato.genome import Genome

EPISODE_DURATION = 200
def run():
    with open('pendulum/neato_pendulum_best_individual_gen74', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()
    for i in range(1):
        env = gym.make("Pendulum-v1") # BipedalWalker-v3
        env.seed(1)
        # env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()
        fitness = 0

        done = False
        while not done:
            action = genome.forward(last_observation)[0]
            next_observation, reward, done, info = env.step([action*2])
            encourage_vertical = 10 * (next_observation[0] + 0.75) ** 2
            discourage_horizontal = 10 * (abs(next_observation[1]) - 1) ** 2
            fitness += (reward + encourage_vertical - discourage_horizontal)
            print(next_observation[0], next_observation[1], encourage_vertical,  discourage_horizontal)
            env.render()
        env.close()
        # print(fitness)

if __name__ == '__main__':
    run()