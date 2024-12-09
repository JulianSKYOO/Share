import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
#import optuna
from typing import Dict, Tuple, List
import logging

class MotionDataset(Dataset):
    def __init__(self, motion_data: Dict[str, np.ndarray], sequence_length: int):
        self.sequence_length = sequence_length
        self.positions = torch.FloatTensor(motion_data['position'])
        self.velocities = torch.FloatTensor(motion_data['velocity'])
        self.trajectories = torch.FloatTensor(motion_data['trajectory'])
        self.custom_features = torch.FloatTensor(motion_data['custom_features']) # I do not know def of custom_features?

    def __len__(self) -> int:
        return len(self.positions) - self.sequence_length

    # I got help from Claude since I do not know the def of motion matching dataset
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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        self.connected = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # You can optimize it with adjusting different layers
        )

    def forward(self, x: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        batch_size = x.size(0)

        if hidden is None:
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (hidden_state, cell_state)

        lstm_out, hidden = self.lstm(x, hidden)
        output = self.connected(lstm_out[:, -1, :])
        return output, hidden

class MotionMatcher:
    def __init__(self, config: dict = None):
        self.config = config or self.default_config()
        self.scaler = StandardScaler()
        self.kd_tree = None
        self.model = None
        self.feature_dims = None
        self.logger = logging.getLogger(__name__)

    # config that you can adjust
    @staticmethod
    def default_config() -> dict:
        return {
            'sequence_length': 30,
            'hidden_size': 32, # this can be used with the power of 2 but typically 8 - 256
            'num_layers': 2,
            'dropout_rate': 0.2, # 0.2 - 0.5 are typical
            'learning_rate': 0.001, # can be adjusted but too high or too low does not guarantee the best predict
            'batch_size': 32, # this also like hidden_size 8 - 32
            'n_candidates': 10, # Nearest poses you can consider
            'prediction_weight': 0.7, # Typically 0.7 you can adjust it with how much you value on LSTM output (higher is more trust)
            'smoothness_weight': 0.3, # you can adjust it, this is to penalise the outliers of the sudden motion changes.
            'max_epochs': 20, # you can adjust it to slightly increase more
            'patience': 10 # early stopping hyperparameter, higher means more tolerance on training.
        }
        # Automating Hyperparameter tuning is available with Optuna

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

        normalized_features = self.scaler.fit_transform(all_features)

        start_idx = 0
        normalized_data = {}
        for key, dim in self.feature_dims.items():
            normalized_data[key] = normalized_features[:, start_idx:start_idx + dim]
            start_idx += dim

        return normalized_data

    # KD-tree for NN search
    def build_kd_tree(self, normalized_data: Dict[str, np.ndarray]):
        feature_matrix = np.concatenate(list(normalized_data.values()), axis=1)
        self.kd_tree = KDTree(feature_matrix)

    # Calculating score for candidate positions based on config
    def compute_pos_score(self, candidate: np.ndarray, lstm_prediction: np.ndarray,
                          current_sequence: np.ndarray) -> float:
        prediction_score = -np.linalg.norm(candidate - lstm_prediction)

        velocity = candidate - current_sequence[-1]
        prev_velocity = current_sequence[-1] - current_sequence[-2]
        acceleration = velocity - prev_velocity
        smoothness_score = -np.linalg.norm(acceleration)

        total_score = (self.config['prediction_weight'] * prediction_score +
                      self.config['smoothness_weight'] * smoothness_score)

        return total_score

    # Best next best position -> this was helped by Claude for distance penalty
    def find_best_next_pos(self, current_sequence: np.ndarray) -> np.ndarray:
        self.model.eval()

        with torch.no_grad():
            normalized_seq = self.scaler.transform(current_sequence)
            x = torch.FloatTensor(normalized_seq).unsqueeze(0)

            lstm_prediction, _ = self.model(x)
            lstm_prediction = lstm_prediction.numpy()[0]

            distances, indices = self.kd_tree.query(
                lstm_prediction.reshape(1, -1),
                k=self.config['n_candidates']
            )

            # Score candidates
            scores = []
            candidates = []
            for idx, distance in zip(indices[0], distances[0]):
                # Extract candidate pose features
                pos_features = self.motion_database['position'][idx]
                vel_features = self.motion_database['velocity'][idx]
                tra_features = self.motion_database['trajectory'][idx]
                cus_features = self.motion_database['custom_features'][idx]

                candidate = np.concatenate([
                    pos_features,
                    vel_features,
                    tra_features,
                    cus_features
                ])

                # Compute score wrt distance penalty
                score = self.compute_pos_score(
                    candidate,
                    lstm_prediction,
                    current_sequence
                )
                distance_penalty = 1.0 / (1.0 + distance)
                final_score = score * distance_penalty

                scores.append(final_score)
                candidates.append(candidate)

            # Select best positions
            best_idx = np.argmax(scores)
            best_pos = candidates[best_idx]

            return best_pos

    def train(self, motion_database: Dict[str, np.ndarray]) -> float:
        
        normalized_data = self.prepare_data(motion_database)
        self.build_kd_tree(normalized_data)

        
        train_data, val_data = train_test_split(normalized_data, test_size=0.2)
        train_dataset = MotionDataset(train_data, self.config['sequence_length'])
        val_dataset = MotionDataset(val_data, self.config['sequence_length'])

        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])

        
        input_dims = sum(self.feature_dims.values())
        self.model = LSTMMotionPredictor(
            input_size=input_dims,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=input_dims,
            dropout_rate=self.config['dropout_rate']
        )

        # Loss function = MSE, Optimizer = ADAM
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['max_epochs']):
            self.model.train()
            train_loss = self._train_epoch(train_loader, optimizer, criterion)

            self.model.eval()
            val_loss = self._validate_epoch(val_loader, criterion)

            self.logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return best_val_loss

    # Training once to calculate the loss
    def _train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> float:
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output, _ = self.model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    # Calculating model performance based on config
    def _validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                output, _ = self.model(batch_x)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    # Generating a sequence for the best positions
    def generate_motion_sequence(self, initial_sequence: np.ndarray,
                               length: int) -> np.ndarray:
        generated_sequence = []
        current_sequence = initial_sequence.copy()

        for _ in range(length):
            next_pos = self.find_best_next_pos(current_sequence)
            generated_sequence.append(next_pos)

            # This is to update the current sequence for the automation
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pos

        return np.array(generated_sequence)


""" # Example of usage of this code
def example_usage():
    n_frames = 1000 # Example for motion data
    n_joints = 23  # Example for humanoid character

    motion_database = {
        'position': np.random.randn(n_frames, n_joints * 3),  # Joint positions (x,y,z)
        'velocity': np.random.randn(n_frames, n_joints * 3),  # Joint velocities (x,y,z)
        'trajectory': np.random.randn(n_frames, 6),  # Root position/orientation
        'custom_features': np.random.randn(n_frames, 10)  # Custom features
    }

    config = {
        'sequence_length': 30,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'n_candidates': 10,
        'prediction_weight': 0.7,
        'smoothness_weight': 0.3,
        'max_epochs': 100,
        'patience': 10
    }

    matcher = MotionMatcher(config)
    matcher.train(motion_database)

    initial_sequence = np.concatenate([
        motion_database['position'][:30],
        motion_database['velocity'][:30],
        motion_database['trajectory'][:30],
        motion_database['custom_features'][:30]
    ], axis=1)

    new_motion = matcher.generate_motion_sequence(initial_sequence, length=60)

    return matcher, new_motion
"""