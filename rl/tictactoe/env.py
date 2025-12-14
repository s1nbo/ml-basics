import gymnasium as gym
import numpy as np


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        # observation space is a 3x3 grid values 0, 1, 2
        # and a current player indicator 0 or 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        # actopm space are the indices 0-8 for the 3x3 grid
        self.action_space = gym.spaces.Discrete(9)

        self.board, _ = self.reset()


    def reset(self):
        super().reset()
        self.current_player = 1
        self.board = np.zeros((9,), dtype=np.int8)
        return self.board.copy(), {}


    def step(self, action):
        if self.board[action] != 0:
            raise ValueError("Invalid action: Cell already occupied")

        self.board[action] = self.current_player
        done = self.check_winner() or np.all(self.board != 0)
        reward = 1 if self.check_winner() else 0

        self.current_player = -1 if self.current_player == 1 else 1

        # return observation, reward, done, truncated, info
        return self.board.copy(), reward, done, False, {}
        

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                return True
        return False
    
    def render(self, mode='human'):
        symbol_map = {1: 'X', -1: 'O', 0: ' '}
        for i in range(3):
            row = [symbol_map[self.board[j]] for j in range(i * 3, (i + 1) * 3)]
            print('|'.join(row))
            if i < 2:
                print('-----')