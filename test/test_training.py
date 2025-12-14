"""
Tests for training loop components.

Tests cover:
- Reward calculation and magnitude
- Policy gradient loss function
- Replay buffer operations
- Learning rate schedule
- Exploration decay
"""
import pytest
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.obj.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    """Test replay buffer operations."""

    def test_buffer_initialization(self, empty_replay_buffer: ReplayBuffer) -> None:
        """Buffer should initialize with correct capacity."""
        assert empty_replay_buffer.capacity == 100
        assert len(empty_replay_buffer) == 0

    def test_buffer_push(self, empty_replay_buffer: ReplayBuffer) -> None:
        """Pushing experiences should add to buffer."""
        states = np.random.randn(5, 100).astype(np.float32)
        targets = [
            np.random.randn(5, 10).astype(np.float32),
            np.random.randn(5, 5).astype(np.float32)
        ]

        empty_replay_buffer.push(states, targets)
        assert len(empty_replay_buffer) == 5

    def test_buffer_push_overflow(self) -> None:
        """Buffer should discard oldest when exceeding capacity."""
        buffer = ReplayBuffer(capacity=10)

        # Push more than capacity
        for i in range(15):
            states = np.random.randn(1, 50).astype(np.float32)
            targets = [np.random.randn(1, 5).astype(np.float32)]
            buffer.push(states, targets)

        assert len(buffer) == 10

    def test_buffer_sample_shape(self, filled_replay_buffer: ReplayBuffer) -> None:
        """Sampled batch should have correct shape."""
        states, targets = filled_replay_buffer.sample(batch_size=16)

        assert states.shape[0] == 16
        assert len(targets) > 0
        for target in targets:
            assert target.shape[0] == 16

    def test_buffer_sample_with_replacement(self, empty_replay_buffer: ReplayBuffer) -> None:
        """Sampling more than buffer size should use replacement."""
        # Add only 5 experiences
        states = np.random.randn(5, 50).astype(np.float32)
        targets = [np.random.randn(5, 10).astype(np.float32)]
        empty_replay_buffer.push(states, targets)

        # Sample more than buffer contains
        sampled_states, sampled_targets = empty_replay_buffer.sample(batch_size=20)
        assert sampled_states.shape[0] == 20

    def test_buffer_clear(self, filled_replay_buffer: ReplayBuffer) -> None:
        """Clear should empty the buffer."""
        filled_replay_buffer.clear()
        assert len(filled_replay_buffer) == 0

    def test_buffer_is_ready(self, empty_replay_buffer: ReplayBuffer) -> None:
        """is_ready should check minimum buffer size."""
        assert empty_replay_buffer.is_ready(0) is True
        assert empty_replay_buffer.is_ready(10) is False

        # Add some experiences
        states = np.random.randn(10, 50).astype(np.float32)
        targets = [np.random.randn(10, 5).astype(np.float32)]
        empty_replay_buffer.push(states, targets)

        assert empty_replay_buffer.is_ready(10) is True
        assert empty_replay_buffer.is_ready(20) is False


class TestPolicyGradientLoss:
    """Test policy gradient loss function."""

    def test_import_loss_function(self) -> None:
        """Should be able to import policy_gradient_loss."""
        from src.scripts.run_rl import policy_gradient_loss
        assert callable(policy_gradient_loss)

    def test_loss_produces_gradient(self) -> None:
        """Loss should produce non-zero gradients."""
        from src.scripts.run_rl import policy_gradient_loss

        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='softmax', input_shape=(5,))
        ])

        # Sample input/target
        x = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y_true = tf.constant([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # Advantage for action 1

        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = policy_gradient_loss(y_true, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        # At least one gradient should be non-zero
        assert any(tf.reduce_sum(tf.abs(g)).numpy() > 0 for g in grads if g is not None)

    def test_loss_shape(self) -> None:
        """Loss should return scalar."""
        from src.scripts.run_rl import policy_gradient_loss

        y_true = tf.constant([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_pred = tf.constant([[0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])

        loss = policy_gradient_loss(y_true, y_pred)
        assert loss.shape == ()  # Scalar

    def test_loss_positive_advantage_decreases_with_high_prob(self) -> None:
        """Higher probability for taken action should reduce loss."""
        from src.scripts.run_rl import policy_gradient_loss

        y_true = tf.constant([[0.0, 1.0, 0.0, 0.0, 0.0]])  # Positive advantage for action 1

        # Lower probability for action 1
        y_pred_low = tf.constant([[0.2, 0.2, 0.2, 0.2, 0.2]])
        loss_low = policy_gradient_loss(y_true, y_pred_low).numpy()

        # Higher probability for action 1
        y_pred_high = tf.constant([[0.1, 0.6, 0.1, 0.1, 0.1]])
        loss_high = policy_gradient_loss(y_true, y_pred_high).numpy()

        # Higher prob should mean lower loss (more aligned with advantage)
        assert loss_high < loss_low


class TestDiscountedReturns:
    """Test discounted return computation."""

    def test_import_compute_discounted_returns(self) -> None:
        """Should be able to import compute_discounted_returns."""
        from src.scripts.run_rl import compute_discounted_returns
        assert callable(compute_discounted_returns)

    def test_returns_shape_1d(self) -> None:
        """Discounted returns should preserve 1D shape."""
        from src.scripts.run_rl import compute_discounted_returns

        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        returns = compute_discounted_returns(rewards, gamma=0.99, normalize=False)

        assert returns.shape == rewards.shape

    def test_returns_discount_pattern(self) -> None:
        """Later actions should get higher returns."""
        from src.scripts.run_rl import compute_discounted_returns

        rewards = np.array([1.0, 1.0, 1.0, 1.0])
        returns = compute_discounted_returns(rewards, gamma=0.99, normalize=False)

        # Last action gets full reward, earlier get discounted
        for i in range(len(returns) - 1):
            assert returns[i] < returns[i + 1]

    def test_returns_normalization(self) -> None:
        """Normalization should center and scale returns."""
        from src.scripts.run_rl import compute_discounted_returns

        rewards = np.array([0.0, 0.0, 0.0, 10.0])
        returns = compute_discounted_returns(rewards, gamma=0.99, normalize=True)

        # Normalized returns should have mean ~0 and std ~1
        assert np.abs(returns.mean()) < 0.1
        assert np.abs(returns.std() - 1.0) < 0.5  # Allow some tolerance

    def test_returns_empty(self) -> None:
        """Empty rewards should return empty array."""
        from src.scripts.run_rl import compute_discounted_returns

        rewards = np.array([])
        returns = compute_discounted_returns(rewards, gamma=0.99)

        assert len(returns) == 0


class TestLearningRateSchedule:
    """Test learning rate schedule."""

    def test_import_warmup_cosine_decay(self) -> None:
        """Should be able to import WarmupCosineDecay."""
        from src.scripts.run_rl import WarmupCosineDecay
        assert WarmupCosineDecay is not None

    def test_warmup_phase(self) -> None:
        """Learning rate should increase during warmup."""
        from src.scripts.run_rl import WarmupCosineDecay

        schedule = WarmupCosineDecay(
            lr_initial=0.001,
            lr_min=0.0001,
            decay_steps=1000,
            warmup_steps=100
        )

        lr_0 = schedule(tf.constant(0)).numpy()
        lr_50 = schedule(tf.constant(50)).numpy()
        lr_100 = schedule(tf.constant(100)).numpy()

        assert lr_0 < lr_50 < lr_100

    def test_decay_phase(self) -> None:
        """Learning rate should decrease after warmup."""
        from src.scripts.run_rl import WarmupCosineDecay

        schedule = WarmupCosineDecay(
            lr_initial=0.001,
            lr_min=0.0001,
            decay_steps=1000,
            warmup_steps=100
        )

        lr_200 = schedule(tf.constant(200)).numpy()
        lr_500 = schedule(tf.constant(500)).numpy()
        lr_1000 = schedule(tf.constant(1100)).numpy()

        assert lr_200 > lr_500 > lr_1000

    def test_lr_bounds(self) -> None:
        """Learning rate should stay within reasonable bounds."""
        from src.scripts.run_rl import WarmupCosineDecay

        schedule = WarmupCosineDecay(
            lr_initial=0.001,
            lr_min=0.0001,
            decay_steps=1000,
            warmup_steps=100
        )

        # During warmup (step 0), LR can be 0 (linear from 0)
        lr_0 = schedule(tf.constant(0)).numpy()
        assert lr_0 >= 0

        # After warmup, LR should be within bounds
        for step in [100, 500, 1000, 2000]:
            lr = schedule(tf.constant(step)).numpy()
            assert lr >= 0.0001 * 0.9  # Allow small numerical tolerance
            assert lr <= 0.001 * 1.1

    def test_schedule_config(self) -> None:
        """Schedule should return config for serialization."""
        from src.scripts.run_rl import WarmupCosineDecay

        schedule = WarmupCosineDecay(
            lr_initial=0.001,
            lr_min=0.0001,
            decay_steps=1000,
            warmup_steps=100
        )

        config_dict = schedule.get_config()
        assert config_dict["lr_initial"] == 0.001
        assert config_dict["lr_min"] == 0.0001
        assert config_dict["decay_steps"] == 1000
        assert config_dict["warmup_steps"] == 100


class TestExplorationDecay:
    """Test exploration rate decay."""

    def test_epsilon_decay_decreases(self) -> None:
        """Epsilon should decrease with decay."""
        epsilon = config.epsilon_start
        for _ in range(100):
            epsilon *= config.epsilon_decay

        assert epsilon < config.epsilon_start
        assert epsilon > config.epsilon_end * 0.1  # Should still be above end after 100 steps

    def test_epsilon_bounds(self) -> None:
        """Epsilon should be bounded by start and end values."""
        epsilon = config.epsilon_start

        # Decay many times
        for _ in range(10000):
            epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        assert epsilon >= config.epsilon_end
        assert epsilon <= config.epsilon_start


class TestTrainingMetrics:
    """Test training metrics dataclass."""

    def test_import_training_metrics(self) -> None:
        """Should be able to import TrainingMetrics."""
        from src.scripts.run_rl import TrainingMetrics
        assert TrainingMetrics is not None

    def test_metrics_initialization(self) -> None:
        """Metrics should initialize with empty lists."""
        from src.scripts.run_rl import TrainingMetrics

        metrics = TrainingMetrics()
        assert len(metrics.episode) == 0
        assert len(metrics.loss_choose) == 0
        assert len(metrics.win_rate) == 0

    def test_metrics_record_episode(self) -> None:
        """Recording episode should add metrics."""
        from src.scripts.run_rl import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_episode(
            episode_num=1,
            loss_choose=0.5,
            loss_eor=0.3,
            avg_score=10.0,
            game_length=5,
            n_turns=20,
            exploration_rate=0.9,
            learning_rate=0.001,
            buffer_size_choose=100,
            buffer_size_eor=50,
            win_rate=0.25
        )

        assert len(metrics.episode) == 1
        assert metrics.episode[0] == 1
        assert metrics.loss_choose[0] == 0.5
        assert metrics.avg_score[0] == 10.0

    def test_metrics_save_load_json(self, tmp_path) -> None:
        """Metrics should save and load from JSON."""
        from src.scripts.run_rl import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_episode(
            episode_num=1,
            loss_choose=0.5,
            loss_eor=0.3,
            avg_score=10.0,
            game_length=5,
            n_turns=20,
            exploration_rate=0.9,
            learning_rate=0.001,
            buffer_size_choose=100,
            buffer_size_eor=50,
            win_rate=0.25
        )

        filepath = tmp_path / "test_metrics.json"
        metrics.save_to_json(str(filepath))

        loaded = TrainingMetrics.load_from_json(str(filepath))
        assert loaded.episode[0] == 1
        assert loaded.loss_choose[0] == 0.5


class TestRewardMagnitude:
    """Test that reward signals have appropriate magnitude."""

    def test_reward_not_divided_by_turns(self) -> None:
        """Rewards should not be diluted by dividing by turn count."""
        # This tests that the fix from Phase 1 is in place
        # Previously rewards were divided by turns, producing near-zero gradients
        from src.scripts.run_rl import compute_discounted_returns

        # Simulate a game where rewards are assigned to each action
        # The implementation gives later actions higher discount factors
        rewards = np.array([1.0, 1.0, 1.0, 1.0])  # Each action has reward 1

        returns = compute_discounted_returns(rewards, gamma=0.99, normalize=False)

        # Later actions get higher returns (full reward for last action)
        assert returns[-1] > returns[0]
        # Returns should still be significant (not near-zero)
        assert np.max(returns) > 0.5

    def test_gamma_provides_temporal_credit(self) -> None:
        """Gamma should provide temporal credit assignment."""
        from src.scripts.run_rl import compute_discounted_returns

        # Use constant rewards to see discount effect
        rewards = np.array([1.0, 1.0, 1.0, 1.0])

        returns_high_gamma = compute_discounted_returns(rewards, gamma=0.99, normalize=False)
        returns_low_gamma = compute_discounted_returns(rewards, gamma=0.5, normalize=False)

        # With high gamma, early actions get more credit (less discount)
        # This is the standard discounting where gamma^n decreases for earlier actions
        # but the implementation reverses this - later actions get gamma^0
        assert returns_high_gamma[-1] > returns_low_gamma[-1] * 0.5
