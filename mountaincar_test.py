import pickle
import gym
import math
import sys

from matplotlib import animation, pyplot as plt
sys.path.append('./neato')
from neato.genome import Genome

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=120)


def run():
    gen = 35
    with open('mountain_car_best_individual', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()

    
    for i in range(5):
        frames = []
        env = gym.make("MountainCarContinuous-v0") # BipedalWalker-v3
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
            frames.append(env.render(mode="rgb_array"))

        env.close()
        save_frames_as_gif(frames, filename=f'mountaincar_{i+1}.gif')


if __name__ == '__main__':
    run()