"""
ADAS Phase 7 - Trajectory Prediction Module
Automotive-grade trajectory prediction with LSTM models and Kalman filtering
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import logging


@dataclass
class TrajectoryState:
    """Vehicle/object state representation"""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    heading: float
    timestamp: float
    confidence: float = 1.0


@dataclass
class PredictionResult:
    """Trajectory prediction result with uncertainty"""
    predicted_states: List[TrajectoryState]
    uncertainty_bounds: np.ndarray
    confidence_score: float
    time_horizon: float


class LSTMTrajectoryPredictor(nn.Module):
    """LSTM-based trajectory prediction for automotive applications"""

    def __init__(self,
                 input_size: int = 6,  # [x, y, vx, vy, ax, ay]
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 prediction_horizon: int = 50,
                 dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        # LSTM backbone for sequence processing
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )

        # Attention mechanism for multi-agent interaction
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

        # Output layers for position and uncertainty
        self.position_head = nn.Linear(hidden_size, 2)  # x, y prediction
        self.uncertainty_head = nn.Linear(hidden_size, 2)  # uncertainty bounds

        # Automotive-specific constraints
        self.max_acceleration = 8.0  # m/s^2
        self.max_velocity = 50.0  # m/s

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with multi-agent interaction modeling

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            attention_mask: Optional mask for multi-agent attention

        Returns:
            Predicted positions and uncertainties
        """
        batch_size, seq_len, _ = x.shape

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)

        # Multi-agent attention for interaction modeling
        if attention_mask is not None:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attention_mask)
            lstm_out = lstm_out + attended_out

        # Predict future trajectory
        predictions = []
        uncertainties = []

        current_hidden = (hidden, cell)
        current_input = x[:, -1:, :]  # Last time step

        for _ in range(self.prediction_horizon):
            lstm_out, current_hidden = self.lstm(current_input, current_hidden)

            # Position prediction with physics constraints
            pos_pred = self.position_head(lstm_out)
            pos_pred = torch.tanh(pos_pred) * self.max_velocity

            # Uncertainty estimation
            uncertainty = torch.softplus(self.uncertainty_head(lstm_out))

            predictions.append(pos_pred)
            uncertainties.append(uncertainty)

            # Update input for next prediction
            current_input = pos_pred

        return torch.cat(predictions, dim=1), torch.cat(uncertainties, dim=1)


class KalmanTrajectoryFilter:
    """Extended Kalman Filter for trajectory smoothing and uncertainty quantification"""

    def __init__(self, dt: float = 0.1):
        self.dt = dt

        # State vector: [x, y, vx, vy, ax, ay]
        self.state_dim = 6

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Observation matrix (position and velocity observable)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.1

        # Measurement noise covariance
        self.R = np.eye(4) * 0.5

        # Initialize state and covariance
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10.0

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step with measurement"""
        # Innovation
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        return self.x, self.P


class MultiAgentInteractionModel:
    """Model multi-agent interactions for trajectory prediction"""

    def __init__(self, interaction_radius: float = 50.0):
        self.interaction_radius = interaction_radius
        self.social_force_strength = 2.0
        self.desired_distance = 5.0

    def compute_social_forces(self,
                            ego_state: TrajectoryState,
                            other_states: List[TrajectoryState]) -> np.ndarray:
        """Compute social forces affecting trajectory"""
        total_force = np.zeros(2)

        for other_state in other_states:
            # Distance and direction
            relative_pos = other_state.position[:2] - ego_state.position[:2]
            distance = np.linalg.norm(relative_pos)

            if distance < self.interaction_radius and distance > 0:
                direction = relative_pos / distance

                # Repulsive force (closer = stronger)
                force_magnitude = self.social_force_strength * np.exp(
                    (self.desired_distance - distance) / self.desired_distance
                )

                # Velocity-dependent adjustment
                relative_vel = other_state.velocity[:2] - ego_state.velocity[:2]
                vel_factor = 1 + 0.5 * np.dot(relative_vel, direction)

                total_force -= direction * force_magnitude * vel_factor

        return total_force


class TrajectoryPredictor:
    """Main trajectory prediction system for ADAS Phase 7"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.logger = logging.getLogger(__name__)

        # Initialize LSTM model
        self.lstm_model = LSTMTrajectoryPredictor().to(device)

        if model_path:
            self.lstm_model.load_state_dict(torch.load(model_path, map_location=device))

        # Initialize Kalman filter for each tracked object
        self.kalman_filters: Dict[int, KalmanTrajectoryFilter] = {}

        # Multi-agent interaction model
        self.interaction_model = MultiAgentInteractionModel()

        # Automotive constraints
        self.max_prediction_time = 5.0  # seconds
        self.min_confidence_threshold = 0.3

    def predict_trajectory(self,
                         object_id: int,
                         history: List[TrajectoryState],
                         other_objects: List[TrajectoryState] = None,
                         prediction_time: float = 3.0) -> PredictionResult:
        """
        Predict trajectory for a single object

        Args:
            object_id: Unique identifier for tracked object
            history: Historical states (minimum 10 time steps)
            other_objects: States of other objects for interaction modeling
            prediction_time: Prediction horizon in seconds

        Returns:
            PredictionResult with predicted states and uncertainties
        """

        if len(history) < 10:
            raise ValueError("Minimum 10 historical states required")

        # Initialize Kalman filter if new object
        if object_id not in self.kalman_filters:
            self.kalman_filters[object_id] = KalmanTrajectoryFilter()

        kf = self.kalman_filters[object_id]

        # Prepare input for LSTM
        input_data = self._prepare_lstm_input(history)

        with torch.no_grad():
            # LSTM prediction
            positions, uncertainties = self.lstm_model(input_data)
            positions = positions.cpu().numpy()[0]
            uncertainties = uncertainties.cpu().numpy()[0]

        # Apply Kalman filtering for smoothing
        predicted_states = []
        dt = 0.1  # 10 Hz prediction
        num_steps = int(prediction_time / dt)

        for i in range(min(num_steps, len(positions))):
            # Kalman prediction
            state, covariance = kf.predict()

            # LSTM measurement update
            measurement = np.array([
                positions[i, 0], positions[i, 1],
                state[2], state[3]  # Use Kalman velocity estimates
            ])

            state, covariance = kf.update(measurement)

            # Apply multi-agent interactions
            if other_objects:
                social_force = self.interaction_model.compute_social_forces(
                    history[-1], other_objects
                )
                # Adjust acceleration based on social forces
                state[4:6] += social_force * 0.1

            # Create predicted state
            predicted_state = TrajectoryState(
                position=state[:2],
                velocity=state[2:4],
                acceleration=state[4:6],
                heading=history[-1].heading,  # TODO: predict heading changes
                timestamp=history[-1].timestamp + (i + 1) * dt,
                confidence=max(0.1, 1.0 - uncertainties[i].mean())
            )

            predicted_states.append(predicted_state)

        # Compute overall confidence
        confidence_score = np.mean([s.confidence for s in predicted_states])

        return PredictionResult(
            predicted_states=predicted_states,
            uncertainty_bounds=uncertainties[:len(predicted_states)],
            confidence_score=confidence_score,
            time_horizon=prediction_time
        )

    def _prepare_lstm_input(self, history: List[TrajectoryState]) -> torch.Tensor:
        """Convert trajectory history to LSTM input format"""

        # Extract features: [x, y, vx, vy, ax, ay]
        features = []
        for state in history[-20:]:  # Use last 20 time steps
            feature_vector = np.concatenate([
                state.position[:2],
                state.velocity[:2],
                state.acceleration[:2]
            ])
            features.append(feature_vector)

        # Pad if necessary
        while len(features) < 20:
            features.insert(0, features[0])

        # Convert to tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        return input_tensor

    def update_model_weights(self,
                           training_data: List[Tuple[List[TrajectoryState], List[TrajectoryState]]]):
        """Online learning update for model adaptation"""

        self.logger.info("Updating trajectory prediction model with new data")

        # Prepare training data
        X, y = [], []
        for history, ground_truth in training_data:
            X.append(self._prepare_lstm_input(history))
            # Convert ground truth to target format
            target = torch.FloatTensor([
                [state.position[0], state.position[1]] for state in ground_truth
            ]).unsqueeze(0).to(self.device)
            y.append(target)

        if len(X) > 0:
            # Simple SGD update (in production, use more sophisticated training)
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            for inputs, targets in zip(X, y):
                optimizer.zero_grad()
                predictions, _ = self.lstm_model(inputs)
                loss = loss_fn(predictions[:, :targets.shape[1], :], targets)
                loss.backward()
                optimizer.step()

    def get_prediction_metrics(self) -> Dict[str, float]:
        """Get current prediction performance metrics"""

        return {
            'active_trackers': len(self.kalman_filters),
            'model_parameters': sum(p.numel() for p in self.lstm_model.parameters()),
            'device': str(self.device),
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }


# Automotive-specific utility functions
def validate_trajectory_safety(prediction: PredictionResult,
                             road_boundaries: np.ndarray,
                             speed_limit: float = 30.0) -> bool:
    """Validate predicted trajectory for automotive safety constraints"""

    for state in prediction.predicted_states:
        # Check speed limits
        speed = np.linalg.norm(state.velocity)
        if speed > speed_limit:
            return False

        # Check road boundaries (simplified)
        if not _point_in_polygon(state.position[:2], road_boundaries):
            return False

        # Check acceleration limits
        acceleration = np.linalg.norm(state.acceleration)
        if acceleration > 8.0:  # m/s^2
            return False

    return True


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if point is inside polygon (road boundaries)"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


if __name__ == "__main__":
    # Example usage for automotive testing
    logging.basicConfig(level=logging.INFO)

    # Create sample trajectory data
    history = []
    for i in range(15):
        state = TrajectoryState(
            position=np.array([i * 2.0, 0.0, 0.0]),
            velocity=np.array([2.0, 0.0, 0.0]),
            acceleration=np.array([0.1, 0.0, 0.0]),
            heading=0.0,
            timestamp=i * 0.1
        )
        history.append(state)

    # Initialize predictor
    predictor = TrajectoryPredictor()

    # Make prediction
    result = predictor.predict_trajectory(
        object_id=1,
        history=history,
        prediction_time=2.0
    )

    print(f"Predicted {len(result.predicted_states)} future states")
    print(f"Confidence score: {result.confidence_score:.3f}")
    print(f"Prediction horizon: {result.time_horizon:.1f}s")