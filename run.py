from marlgrid import envs
import gym
import numpy as np


def main():
    env = gym.make('MarlGrid-2AgentComms15x15-v0')
    env.reset()

    obs_list = env.reset()

    done = False
    while not done:
        env.render()  # OPTIONAL: render the whole scene + birds eye view
        agent_actions = np.random.randint(env.action_space[0].n, size=len(env.action_space))
        next_obs_list, rew_list, done, _ = env.step(agent_actions)
        print(rew_list)

if __name__ == '__main__':
    main()
