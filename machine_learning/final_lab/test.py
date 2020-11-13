from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

levels = {  0: 'SuperMarioBros-1-1-v0',
            1: 'SuperMarioBros-1-2-v0',
            2: 'SuperMarioBros-1-4-v0',
            3: 'SuperMarioBros-2-2-v0',
            4: 'SuperMarioBros-3-1-v0',
            5: 'SuperMarioBros-6-3-v0',
}

env = gym_super_mario_bros.make(levels[0])
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()