from env import TicTacToeEnv
from agent import RlAgent, RandomAgent

env = TicTacToeEnv()
agent1 = RandomAgent("RandomAgent1")
agent2 = RandomAgent("RlAgent2")

obs, _ = env.reset()
done = False

env.render()

while not done:
    if env.current_player == 1:
        action = agent1.get_action(obs)
    else:
        action = agent2.get_action(obs * -1)

    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        if reward == 1:
            print("Player 1 wins")
        elif reward == -1:
            print("Player 2 wins")
        else:
            print("Draw")
        break
