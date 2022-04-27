import pickle
import gym
import sys
sys.path.append('./neato')
from neato.genome import Genome

EPISODE_DURATION = 200
def run():
    with open(r'pendulum\models\neato_pendulum_59_898', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()
    for i in range(5):
        env = gym.make("Pendulum-v1") # BipedalWalker-v3
        env.seed(69)
        # env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()
        fitness = 0

        done = False
        while not done:
            action = genome.forward(last_observation)[0]
            next_observation, reward, done, info = env.step([action*2])
            encourage_vertical = (next_observation[0] + 0.75) ** 2
            discourage_horizontal = (abs(next_observation[1]) - 1) ** 2
            discourage_speed = action ** 2
            fitness += (reward + encourage_vertical - discourage_horizontal - discourage_speed)
            # print(action * 2)
            last_observation = next_observation
            env.render()
        env.close()
        print(fitness)

if __name__ == '__main__':
    run()