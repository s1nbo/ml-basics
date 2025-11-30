import os
from typing import List

import numpy as np
import torch
from torch.distributions import Categorical

from game import SnakeGameAI, BLOCK_SIZE
from model import ActorCritic, device

# ---------- Hyperparameters ----------

N_ENVS = 64
STEPS_PER_UPDATE = 128  # rollout length per env
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 2048

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LR = 2.5e-4


class PPOAgent:
    def __init__(self, grid_height: int, grid_width: int, n_actions: int = 3):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_channels = 3  # head, body, food
        self.input_dim = self.n_channels * self.grid_height * self.grid_width
        self.n_actions = n_actions

        self.model = ActorCritic(self.input_dim, self.n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        self.global_step = 0
        self.global_episode = 0
        self.best_score = 0

    # --------- State encoding: full grid ---------

    def encode_state(self, game: SnakeGameAI) -> np.ndarray:
        """Encode full board as a (C, H, W) grid."""
        grid = np.zeros(
            (self.n_channels, self.grid_height, self.grid_width), dtype=np.float32
        )

        # Snake body (including head)
        for idx, pt in enumerate(game.snake):
            gx = pt.x // BLOCK_SIZE
            gy = pt.y // BLOCK_SIZE
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                if idx == 0:
                    grid[0, gy, gx] = 1.0  # head
                else:
                    grid[1, gy, gx] = 1.0  # body

        # Food
        fx = game.food.x // BLOCK_SIZE
        fy = game.food.y // BLOCK_SIZE
        if 0 <= fx < self.grid_width and 0 <= fy < self.grid_height:
            grid[2, fy, fx] = 1.0

        return grid

    # --------- PPO helpers ---------

    def select_action(self, state_batch: torch.Tensor):
        """Given a batch of states [N, input_dim], sample actions and get logprobs & values."""
        logits, values = self.model(state_batch)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)

    def compute_gae(self, rewards, dones, values, next_values):
        """
        rewards, dones, values: [T, N]
        next_values: [N]
        returns advantages, returns: [T, N]
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards, device=device)
        gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # 1 for non-terminal, 0 for terminal
            delta = rewards[t] + GAMMA * next_values * mask - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages[t] = gae
            next_values = values[t]

        returns = advantages + values
        return advantages, returns

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        """
        states: [T*N, input_dim]
        actions, log_probs, returns, advantages: [T*N]
        """
        batch_size = states.size(0)
        indices = np.arange(batch_size)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, batch_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idx = indices[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logits, values = self.model(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = torch.nn.functional.mse_loss(
                    values.squeeze(-1), mb_returns
                )

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def save(self, path: str = "./model/ppo_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "./model/ppo_model.pth"):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded PPO model from {path}")
        else:
            print(f"No PPO model found at {path}, starting fresh.")


# ---------- PPO Multi-env training ----------


def train_ppo_multi_env():
    # Create a temporary game to infer grid size
    tmp_game = SnakeGameAI(visualize=False)
    grid_w = tmp_game.w // BLOCK_SIZE
    grid_h = tmp_game.h // BLOCK_SIZE
    del tmp_game

    agent = PPOAgent(grid_height=grid_h, grid_width=grid_w, n_actions=3)
    agent.load()

    # Create environments
    envs: List[SnakeGameAI] = [SnakeGameAI(visualize=False) for _ in range(N_ENVS)]

    # Initial states
    states_np = np.stack([agent.encode_state(env) for env in envs])  # [N, C, H, W]

    while True:
        rollout_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        for _ in range(STEPS_PER_UPDATE):
            # Flatten states for the network: [N, input_dim]
            flat_states = states_np.reshape(N_ENVS, -1)
            state_batch = torch.tensor(flat_states, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, log_probs, values = agent.select_action(state_batch)

            # Step environments
            rewards = np.zeros(N_ENVS, dtype=np.float32)
            dones = np.zeros(N_ENVS, dtype=np.float32)

            new_states_np = np.zeros_like(states_np)
            for i, env in enumerate(envs):
                # Map discrete action to Snake move: 0=straight,1=right,2=left
                action_idx = int(actions[i].item())
                move = [0, 0, 0]
                move[action_idx] = 1

                reward, done, score = env.play_step(move)
                rewards[i] = reward
                dones[i] = float(done)

                if done:
                    agent.global_episode += 1
                    if score > agent.best_score:
                        agent.best_score = score
                        agent.save()
                    print(
                        f"Episode {agent.global_episode} | "
                        f"Score: {score} | Best: {agent.best_score}"
                    )
                    env.reset()
                    new_states_np[i] = agent.encode_state(env)
                else:
                    new_states_np[i] = agent.encode_state(env)

            # Store rollout data
            rollout_states.append(state_batch)
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)
            rollout_values.append(values)
            rollout_rewards.append(
                torch.tensor(rewards, dtype=torch.float32, device=device)
            )
            rollout_dones.append(
                torch.tensor(dones, dtype=torch.float32, device=device)
            )

            states_np = new_states_np
            agent.global_step += N_ENVS

        # Convert rollout to tensors [T, N, ...]
        states = torch.stack(rollout_states)           # [T, N, input_dim]
        actions = torch.stack(rollout_actions)         # [T, N]
        log_probs = torch.stack(rollout_log_probs)     # [T, N]
        values = torch.stack(rollout_values)           # [T, N]
        rewards = torch.stack(rollout_rewards)         # [T, N]
        dones = torch.stack(rollout_dones)             # [T, N]

        # Compute next_values from last states
        flat_last_states = states_np.reshape(N_ENVS, -1)
        with torch.no_grad():
            _, next_values = agent.model(
                torch.tensor(flat_last_states, dtype=torch.float32, device=device)
            )
        next_values = next_values.squeeze(-1)  # [N]

        # Compute advantages & returns
        advantages, returns = agent.compute_gae(
            rewards=rewards, dones=dones, values=values, next_values=next_values
        )

        # Flatten [T, N] -> [T*N]
        T, N = rewards.shape
        flat_states_all = states.view(T * N, -1)
        flat_actions = actions.reshape(T * N)
        flat_log_probs = log_probs.reshape(T * N)
        flat_returns = returns.reshape(T * N)
        flat_advantages = advantages.reshape(T * N)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        # PPO update
        agent.ppo_update(
            states=flat_states_all,
            actions=flat_actions,
            log_probs=flat_log_probs,
            returns=flat_returns,
            advantages=flat_advantages,
        )


# ---------- Play with trained PPO policy ----------


def play_ppo_model():
    # Create a temporary game to infer grid size
    tmp_game = SnakeGameAI(visualize=True)
    grid_w = tmp_game.w // BLOCK_SIZE
    grid_h = tmp_game.h // BLOCK_SIZE

    agent = PPOAgent(grid_height=grid_h, grid_width=grid_w, n_actions=3)
    agent.load()

    env = tmp_game  # already visualize=True

    while True:
        state_grid = agent.encode_state(env)
        flat_state = state_grid.reshape(1, -1)
        state_tensor = torch.tensor(flat_state, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, _ = agent.model(state_tensor)
            dist = Categorical(logits=logits)
            # Greedy action (no exploration) for playing
            action = torch.argmax(dist.probs, dim=-1).item()

        move = [0, 0, 0]
        move[action] = 1

        _, done, score = env.play_step(move)

        if done:
            print(f"Game over! Score: {score}")
            env.reset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "play"],
        default="train",
        help="train: PPO multi-env training; play: watch the PPO policy play Snake",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_ppo_multi_env()
    else:
        play_ppo_model()
