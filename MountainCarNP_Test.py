import os
import pickle
import neat
import gym 
import numpy as np
import math
from matplotlib import animation, pyplot as plt

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

    with open('MountainCar_winner', 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configMountainCar')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)


    for i in range(5):
        frames = []
        env = gym.make("MountainCarContinuous-v0") 
        last_observation = env.reset()

        fitness = 0.0
        done = False
        while not done:
            action = net.activate(last_observation)

            next_observation, reward, done, info = env.step(action)
            reward = reward = 100*((math.sin(3*next_observation[0]) * 0.0025 + 0.5 * next_observation[1] * next_observation[1]) - (math.sin(3*last_observation[0]) * 0.0025 + 0.5 * last_observation[1] * last_observation[1]))
            last_observation = next_observation
            fitness += reward
            env.render()
            frames.append(env.render(mode="rgb_array"))

        env.close()
        save_frames_as_gif(frames, filename=f'mountaincar_NeatPython_{i+1}.gif')


if __name__ == '__main__':
    run()
