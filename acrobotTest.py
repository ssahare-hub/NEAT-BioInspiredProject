import pickle
import gym
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
    anim.save(path + filename, writer='imagemagick', fps=60)

EPISODE_DURATION = 200
def run():
    with open(r'acrobat\models\neato_acrobat_89_884', 'rb') as f:
        genome: Genome = pickle.load(f)

    print('Loaded genome:')
    for n in genome._connections:
        genome._connections[n].show_connections()
    for i in range(5):
        env = gym.make("Acrobot-v1") # BipedalWalker-v3
        # env.seed(1)
        frames = []
        # env._max_episode_steps = EPISODE_DURATION
        last_observation = env.reset()
        fitness = 0

        done = False
        while not done:
            output = genome.forward(last_observation)[0]
            if output <= 0.3:
                action = -1
            elif 0.3 < output <= 0.6:
                action = 0
            else:
                action = 1
            next_observation, reward, done, info = env.step(action)
            last_observation = next_observation
            fitness += reward
            env.render()
            frames.append(env.render(mode="rgb_array"))
        env.close()
        save_frames_as_gif(frames, filename=f'acrobot_{i+1}.gif')
        # print(fitness)

if __name__ == '__main__':
    run()