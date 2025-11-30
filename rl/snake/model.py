import os
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Simple MLP-based Actor-Critic network.
    Input: flat grid vector of size input_dim
    Outputs:
      - policy logits over actions
      - state value estimate
    """

    def __init__(self, input_dim: int, n_actions: int, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, n_actions)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: [B, input_dim]
        feat = self.shared(x)
        logits = self.policy_head(feat)
        values = self.value_head(feat)
        return logits, values

    def save(self, file_name: str = "ppo_model.pth") -> None:
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str = "ppo_model.pth") -> None:
        file_path = os.path.join("./model", file_name)
        if os.path.exists(file_path):
            state_dict = torch.load(file_path, map_location=device)
            self.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model found at {file_path}")
