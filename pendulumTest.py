import pickle

#import cart_pole
import gym
import sys

sys.path.append('./neato')

EPISODE_DURATION = 750
def run():
    with open('pendulum_best_individual_gen61', 'rb') as f:
        genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].showConn()

    env = gym.make("Pendulum-v1") # BipedalWalker-v3
    env._max_episode_steps = EPISODE_DURATION
    observation = env.reset()
    fitness = 0

    done = False
    while not done:
        action = genome.forward(observation)[0]

        observation, reward, done, info = env.step([action*2])
        fitness += reward
        env.render()

    print(fitness+0*EPISODE_DURATION)

if __name__ == '__main__':
    run()