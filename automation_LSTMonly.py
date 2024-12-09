import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

class MotionDataset(Dataset):
    def __init__(self, motion_data: Dict[str, np.ndarray], sequence_length: int):
        self.sequence_length = sequence_length
        self.positions = torch.FloatTensor(motion_data['position'])
        self.velocities = torch.FloatTensor(motion_data['velocity'])
        self.trajectories = torch.FloatTensor(motion_data['trajectory'])
        self.custom_features = torch.FloatTensor(motion_data['custom_features'])

    def __len__(self) -> int:
        return len(self.positions) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_seq = self.positions[idx:idx + self.sequence_length]
        vel_seq = self.velocities[idx:idx + self.sequence_length]
        traj_seq = self.trajectories[idx:idx + self.sequence_length]
        custom_seq = self.custom_features[idx:idx + self.sequence_length]

        target_pos = self.positions[idx + self.sequence_length]
        target_vel = self.velocities[idx + self.sequence_length]
        target_traj = self.trajectories[idx + self.sequence_length]
        target_custom = self.custom_features[idx + self.sequence_length]

        x = torch.cat([pos_seq, vel_seq, traj_seq, custom_seq], dim=1)
        y = torch.cat([target_pos, target_vel, target_traj, target_custom])

        return x, y

class LSTMMotionPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.connected = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (hidden_state, cell_state))

        # Last output
        output = self.connected(lstm_out[:, -1, :])
        return output

class LSTMMotionMatcher:
    def __init__(self, config: dict = None):
        self.config = config or {
            'sequence_length': 30,
            'hidden_size': 256,
            'num_layers': 2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_candidates': 10,
            'prediction_weight': 0.7,
            'smoothness_weight': 0.3,
            'max_epochs': 100,
            'patience': 10
        }
        self.scaler = StandardScaler()
        self.model = None

    def prepare_data(self, motion_database: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.motion_database = motion_database

        all_features = np.concatenate([
            motion_database['position'],
            motion_database['velocity'],
            motion_database['trajectory'],
            motion_database['custom_features']
        ], axis=1)

        self.feature_dims = {
            'position': motion_database['position'].shape[1],
            'velocity': motion_database['velocity'].shape[1],
            'trajectory': motion_database['trajectory'].shape[1],
            'custom': motion_database['custom_features'].shape[1]
        }

        return self.scaler.fit_transform(all_features)

    def compute_pos_score(self, candidate: np.ndarray, prediction: np.ndarray,
                          current_sequence: np.ndarray) -> float:
        prediction_distance = -np.linalg.norm(candidate - prediction[:len(candidate)])

        velocity = candidate - current_sequence[-1, :len(candidate)]
        prev_velocity = current_sequence[-1, :len(candidate)] - current_sequence[-2, :len(candidate)]
        acceleration = velocity - prev_velocity
        smoothness_score = -np.linalg.norm(acceleration)

        total_score = (self.config['prediction_weight'] * prediction_distance +
                      self.config['smoothness_weight'] * smoothness_score)

        return total_score

    def train(self, motion_database: Dict[str, np.ndarray]) -> float:
        normalized_data = self.prepare_data(motion_database)

        dataset = MotionDataset(normalized_data, self.config['sequence_length'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        input_dims = sum(self.feature_dims.values())
        self.model = LSTMMotionPredictor(
            input_size=input_dims,
            hidden_size=self.config['hidden_size'],
            output_size=input_dims,
            num_layers=self.config['num_layers']
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['max_epochs']):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print("Early stopped")
                    break

        return best_loss

    def find_best_next_pos(self, current_sequence: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            normalized_seq = self.scaler.transform(current_sequence)
            x = torch.FloatTensor(normalized_seq).unsqueeze(0)

            prediction = self.model(x).numpy()[0]

            best_score = float('-inf')
            best_pos = None

            pos_dim = self.feature_dims['position']

            for i in range(len(self.motion_database['position'])):
                candidate = self.motion_database['position'][i]
                score = self.compute_pos_score(candidate, prediction[:pos_dim], current_sequence)

                if score > best_score:
                    best_score = score
                    best_pos = candidate

            return best_pos

    def generate_motion_sequence(self, initial_sequence: np.ndarray, length: int) -> np.ndarray:
        generated_sequence = []
        current_sequence = initial_sequence.copy()

        for _ in range(length):
            next_pos = self.find_best_next_pos(current_sequence)
            generated_sequence.append(next_pos)

            # Update current
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pos

        return np.array(generated_sequence)

"""
# Example usage of this
def example_usage():
    # Create sample motion data
    n_frames = 1000
    n_joints = 23

    motion_database = {
        'position': np.random.randn(n_frames, n_joints * 3),
        'velocity': np.random.randn(n_frames, n_joints * 3),
        'trajectory': np.random.randn(n_frames, 6),
        'custom_features': np.random.randn(n_frames, 10)
    }

    config = {
        'sequence_length': 30,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'n_candidates': 10,
        'prediction_weight': 0.7,
        'smoothness_weight': 0.3,
        'max_epochs': 20,
        'patience': 10
    }

    matcher = LSTMMotionMatcher(config)
    matcher.train(motion_database)

    initial_sequence = np.concatenate([
        motion_database['position'][:30],
        motion_database['velocity'][:30],
        motion_database['trajectory'][:30],
        motion_database['custom_features'][:30]
    ], axis=1)

    new_motion = matcher.generate_motion_sequence(initial_sequence, length=60)

    print(f"Generated motion shape: {new_motion.shape}")
    return matcher, new_motion
"""
    
if __name__ == "__main__":
    matcher, generated_motion = example_usage()