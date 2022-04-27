import pickle
import gym
import sys
sys.path.append('./neato')
from neato.genome import Genome
from matplotlib import animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def run():
    with open('cartpole/neato_cartpole_best_individual_gen2', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()

    env = gym.make("CartPole-v1") # BipedalWalker-v3
    observation = env.reset()
    frames = []
    done = False
    while not done:
        frames.append(env.render(mode="rgb_array"))
        action = genome.forward(observation)[0]

        observation, reward, done, info = env.step(action <= 0.5)
        env.render()
    env.close()
    save_frames_as_gif(frames)

if __name__ == '__main__':
    run()