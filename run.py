from marlgrid import envs
import gym
import numpy as np
from marlgrid.envs import env_from_config

# from marlgrid.envs.empty import EmptyColorMultiGrid
from marlgrid.envs.hallways import HallWaysMultiGrid


def main():
    env = gym.make(
        "MarlGrid-2AgentComms15x15-v0"
    )  # ('MarlGrid-2AgentEmptyColor15x15-v0')

    env.reset()

    obs_list = env.reset()
    env.render()
    done = False
    steps = 0
    # actions = [[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[0,5],[5,2],
    #            [2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[0,0],
    #            [2,2],[2,2],[2,2],[2,2],[2,2],[5,2],[2,5],[5,5]]
    # actions = [[0,0],[0,0],[2,0], [0,0], [2,0], [0,0],[0,2],[2,0]]
    actions = [
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 0],
        [0, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 0],
    ]

    actions = [
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [0, 0],
        [2, 2],
        [2, 0],
    ]
    # while not done:
    for action in actions:
        # OPTIONAL: render the whole scene + birds eye view
        # agent_actions = np.random.randint(env.action_space[0].n, size=len(env.action_space))
        next_obs_list, rew_list, done, _ = env.step(action)
        steps += 1
        env.render()
        print(steps, action, rew_list, done)
        print()


def main2():
    env = gym.make(
        "MarlGrid-2AgentComms15x15-v0"
    )  # 'MarlGrid-2AgentComms15x15-v0')  # ('MarlGrid-2AgentEmptyColor15x15-v0')
    env = gym.wrappers.Monitor(
        env,
        "./recording",
        video_callable=lambda episode_id: episode_id == 0,
        force=True,
    )
    env.seed(55)
    env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

    print("wait")


def main3():
    env_config = {
        "env_class": "HallWaysMultiGrid",  ##"EmptyColorMultiGrid",
        "grid_size": 15,
        "goal_colors": ["blue", "red", "blue", "red"],
        "goal_coordinates": [[7, 1], [8, 1], [8, 13], [7, 13]],
        "max_steps": 250,
        "respawn": False,
        "ghost_mode": False,
        "reward_decay": False,
    }
    player_interface_config = {
        "view_size": 7,
        "view_offset": 1,
        "view_tile_size": 1,
        "observation_style": "rich",
        "see_through_walls": False,
        "observe_position": True,
        "see_color_in_view_bool": True,
        "observe_orientation": True,
        "color": "orange",
    }
    p1 = dict(player_interface_config.copy())
    p2 = dict(player_interface_config.copy())
    p1["color"] = "orange"
    p2["color"] = "yellow"
    env_config["agents"] = [p1, p2]
    env = env_from_config(env_config)
    obs = env.reset()
    env.render()
    print(obs[0])


def main4():
    # env = gym.make("MarlGrid-2AgentCommGame15x15-v0")
    env_config = {
        "env_class": "CommunicationGameEnv",
        "grid_size": 15,
        "respawn": False,
        "ghost_mode": False,
        "reward_decay": False,
        "block_coordinates": [(1, 1), (13, 1), (1, 13), (13, 13)],
        "block_colors": ["blue", "red", "cyan", "pink"],
        "comm_blocks_coordinates": [(7, 4), (7, 10)],
        "max_steps": 250,
    }
    player_interface_config = {
        "view_size": 7,
        "view_offset": 1,
        "view_tile_size": 1,
        "observation_style": "rich",
        "see_through_walls": False,
        "observe_position": True,
        "see_color_in_view_bool": True,
        "observe_orientation": True,
        "color": "green",
    }
    p1 = dict(player_interface_config.copy())
    p2 = dict(player_interface_config.copy())
    # p1["color"] = "orange"
    # p2["color"] = "yellow"
    env_config["agents"] = [p1, p2]
    env = env_from_config(env_config)

    env.seed(22)
    env.reset()
    print("starting")
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        print('obs:', obs[0]['see_color_in_view'], obs[1]['see_color_in_view'])
        if done:
            break


if __name__ == "__main__":
    main4()
