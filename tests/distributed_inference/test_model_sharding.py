"""Test suite for ModelShardingEngine

Tests partitioning algorithms, device configurations, memory constraints,
and activation passing between shards.
"""

import gc
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
from packages.core.resources.device_profiler import DeviceProfiler
from packages.core.resources.resource_monitor import ResourceMonitor
from src.production.distributed_inference.model_sharding_engine import (
    DeviceProfile,
    ModelShard,
    ModelShardingEngine,
    ShardingPlan,
    ShardingStrategy,
)

from packages.p2p.core.p2p_node import P2PNode, PeerCapabilities


@pytest.fixture
def mock_p2p_node():
    """Mock P2P node for testing"""
    node = AsyncMock(spec=P2PNode)
    node.node_id = "test_node_1"
    node.peer_registry = {}
    node.local_capabilities = PeerCapabilities(
        device_id="test_node_1",
        cpu_cores=4,
        ram_mb=8192,
        battery_percent=80,
        trust_score=0.9,
        evolution_capacity=0.8,
    )
    node.get_suitable_evolution_peers.return_value = []
    node.broadcast_to_peers = AsyncMock(return_value=1)
    return node


@pytest.fixture
def mock_resource_monitor():
    """Mock resource monitor for testing"""
    monitor = AsyncMock(spec=ResourceMonitor)
    return monitor


@pytest.fixture
def mock_device_profiler():
    """Mock device profiler for testing"""
    profiler = AsyncMock(spec=DeviceProfiler)
    profiler.current_snapshot = None
    return profiler


@pytest.fixture
def device_profiles():
    """Sample device profiles for testing"""
    return [
        DeviceProfile(
            device_id="device_1",
            capabilities=PeerCapabilities(
                device_id="device_1",
                cpu_cores=8,
                ram_mb=16384,
                trust_score=0.9,
                evolution_capacity=0.9,
            ),
            available_memory_mb=12288,  # 75% of 16GB
            compute_score=10.0,
            network_latency_ms=10.0,
            reliability_score=0.9,
        ),
        DeviceProfile(
            device_id="device_2",
            capabilities=PeerCapabilities(
                device_id="device_2",
                cpu_cores=4,
                ram_mb=8192,
                trust_score=0.8,
                evolution_capacity=0.7,
            ),
            available_memory_mb=6144,  # 75% of 8GB
            compute_score=6.0,
            network_latency_ms=20.0,
            reliability_score=0.8,
        ),
        DeviceProfile(
            device_id="device_3",
            capabilities=PeerCapabilities(
                device_id="device_3",
                cpu_cores=2,
                ram_mb=4096,
                trust_score=0.7,
                evolution_capacity=0.6,
            ),
            available_memory_mb=3072,  # 75% of 4GB
            compute_score=3.0,
            network_latency_ms=50.0,
            reliability_score=0.7,
        ),
    ]


@pytest.fixture
def sharding_engine(mock_p2p_node, mock_resource_monitor, mock_device_profiler):
    """Create ModelShardingEngine for testing"""
    return ModelShardingEngine(mock_p2p_node, mock_resource_monitor, mock_device_profiler)


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestModelShardingEngine:
    """Test cases for ModelShardingEngine"""

    @pytest.mark.asyncio
    async def test_initialization(self, sharding_engine):
        """Test proper initialization of sharding engine"""
        assert sharding_engine.current_sharding_plan is None
        assert len(sharding_engine.active_shards) == 0
        assert len(sharding_engine.device_assignments) == 0
        assert sharding_engine.stats["models_sharded"] == 0

    @pytest.mark.asyncio
    async def test_analyze_model_structure(self, sharding_engine, temp_model_dir):
        """Test model structure analysis"""
        model_path = temp_model_dir

        # Mock model configuration
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_loader,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        ):
            # Mock tokenizer
            mock_tokenizer.return_value = MagicMock()

            # Mock model config
            mock_config = MagicMock()
            mock_config.num_hidden_layers = 12
            mock_config.hidden_size = 768
            mock_config.num_attention_heads = 12
            mock_config.vocab_size = 30000
            mock_config.intermediate_size = 3072

            mock_config_loader.return_value = mock_config
            mock_model.side_effect = AssertionError("Model weights should not be loaded")

            # Test analysis
            analysis = await sharding_engine._analyze_model(model_path)

            assert analysis["model_path"] == model_path
            assert analysis["num_layers"] == 12
            assert analysis["hidden_size"] == 768
            assert analysis["num_attention_heads"] == 12
            assert analysis["vocab_size"] == 30000
            assert "layer_memory_mb" in analysis
            assert "total_memory_mb" in analysis
            assert "embedding_memory_mb" in analysis
            assert analysis["can_split_attention"] is True  # 12 heads >= 4

    @pytest.mark.asyncio
    async def test_model_analysis_memory_usage(self, sharding_engine):
        """Ensure model analysis doesn't increase memory excessively"""
        sharding_engine.model_analysis_cache.clear()
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_loader,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        ):
            mock_tokenizer.return_value = MagicMock()

            mock_cfg = MagicMock()
            mock_cfg.num_hidden_layers = 12
            mock_cfg.hidden_size = 768
            mock_cfg.num_attention_heads = 12
            mock_cfg.vocab_size = 30000
            mock_cfg.intermediate_size = 3072

            mock_config_loader.return_value = mock_cfg

            def heavy_model(*args, **kwargs):
                return MagicMock(buffer=bytearray(150 * 1024 * 1024))

            mock_model.side_effect = heavy_model

            gc.collect()
            process = psutil.Process()
            before = process.memory_info().rss
            await sharding_engine._analyze_model("unique-model-path")
            after = process.memory_info().rss

            assert not mock_model.called
            assert after - before < 100 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_get_device_profiles(self, sharding_engine, device_profiles):
        """Test device profile generation"""
        # Mock peer registry
        sharding_engine.p2p_node.peer_registry = {
            "device_1": device_profiles[0].capabilities,
            "device_2": device_profiles[1].capabilities,
            "device_3": device_profiles[2].capabilities,
        }

        # Mock local capabilities
        sharding_engine.p2p_node.local_capabilities = device_profiles[0].capabilities
        sharding_engine.p2p_node.get_suitable_evolution_peers.return_value = [
            device_profiles[1].capabilities,
            device_profiles[2].capabilities,
        ]

        profiles = await sharding_engine._get_device_profiles()

        assert len(profiles) >= 2  # At least 2 suitable devices
        # Should be sorted by suitability (descending)
        assert profiles[0].available_memory_mb >= profiles[1].available_memory_mb

    @pytest.mark.asyncio
    async def test_mixed_memory_devices(self, sharding_engine, temp_model_dir):
        """Ensure low-memory devices are penalized and receive smaller shards."""

        high_peer = PeerCapabilities(
            device_id="high_mem",
            cpu_cores=4,
            ram_mb=8192,
            trust_score=0.9,
            evolution_capacity=0.9,
        )
        medium_peer = PeerCapabilities(
            device_id="med_mem",
            cpu_cores=4,
            ram_mb=8192,
            trust_score=0.9,
            evolution_capacity=0.9,
        )
        low_peer = PeerCapabilities(
            device_id="low_mem",
            cpu_cores=16,  # High compute but low memory
            ram_mb=2048,
            trust_score=0.9,
            evolution_capacity=0.9,
        )

        sharding_engine.p2p_node.local_capabilities = high_peer
        sharding_engine.p2p_node.node_id = "high_mem"
        sharding_engine.p2p_node.get_suitable_evolution_peers.return_value = [
            medium_peer,
            low_peer,
        ]

        profiles = await sharding_engine._get_device_profiles()

        low_profile = next(p for p in profiles if p.device_id == "low_mem")
        baseline_compute = (low_peer.cpu_cores * 2 + low_peer.ram_mb / 1024) / 10
        assert low_profile.compute_score == pytest.approx(baseline_compute * 0.5)

        mock_analysis = {
            "model_path": temp_model_dir,
            "num_layers": 7,
            "layer_memory_mb": 100.0,
            "layer_compute_score": 1.0,
            "total_memory_mb": 700.0,
        }

        plan = await sharding_engine._create_memory_aware_plan(mock_analysis, profiles)
        low_shards = [s for s in plan.shards if s.device_id == "low_mem"]
        assert low_shards  # low-memory device should receive a shard
        assert all(len(s.layer_indices) == 1 for s in low_shards)
        other_shards = [s for s in plan.shards if s.device_id != "low_mem"]
        assert any(len(s.layer_indices) > 1 for s in other_shards)

    @pytest.mark.asyncio
    async def test_memory_aware_sharding(self, sharding_engine, device_profiles, temp_model_dir):
        """Test memory-aware sharding strategy"""
        # Mock model analysis
        mock_analysis = {
            "model_path": temp_model_dir,
            "num_layers": 12,
            "layer_memory_mb": 100.0,  # 100MB per layer
            "layer_compute_score": 1.0,
            "total_memory_mb": 1200.0,
        }

        with patch.object(sharding_engine, "_analyze_model", return_value=mock_analysis):
            plan = await sharding_engine._create_memory_aware_plan(mock_analysis, device_profiles)

            # Verify plan structure
            assert isinstance(plan, ShardingPlan)
            assert plan.total_shards > 0
            assert len(plan.shards) == plan.total_shards

            # Verify memory constraints are respected
            for shard in plan.shards:
                device = next(d for d in device_profiles if d.device_id == shard.device_id)
                assert shard.memory_mb <= device.available_memory_mb

            # Verify activation routing
            assert len(plan.activation_routing) == len(plan.shards)

    @pytest.mark.asyncio
    async def test_compute_balanced_sharding(self, sharding_engine, device_profiles, temp_model_dir):
        """Test compute-balanced sharding strategy"""
        mock_analysis = {
            "model_path": temp_model_dir,
            "num_layers": 12,
            "layer_memory_mb": 80.0,
            "layer_compute_score": 1.0,
            "total_memory_mb": 960.0,
        }

        with patch.object(sharding_engine, "_analyze_model", return_value=mock_analysis):
            plan = await sharding_engine._create_compute_balanced_plan(mock_analysis, device_profiles)

            # Verify plan structure
            assert isinstance(plan, ShardingPlan)
            assert plan.total_shards > 0

            # Verify compute distribution is reasonable
            device_loads = {}
            for shard in plan.shards:
                device = next(d for d in device_profiles if d.device_id == shard.device_id)
                load_ratio = shard.compute_requirement / device.compute_score
                device_loads[shard.device_id] = device_loads.get(shard.device_id, 0) + load_ratio

            # Should have relatively balanced loads
            load_values = list(device_loads.values())
            if len(load_values) > 1:
                load_variance = sum((x - sum(load_values) / len(load_values)) ** 2 for x in load_values)
                assert load_variance < 2.0  # Reasonable variance threshold

    @pytest.mark.asyncio
    async def test_sharding_plan_optimization(self, sharding_engine, device_profiles, temp_model_dir):
        """Test sharding plan optimization"""
        # Create a plan with memory constraint violations
        problematic_shard = ModelShard(
            shard_id="problem_shard",
            device_id="device_3",  # Small device
            layer_indices=[0, 1, 2, 3, 4],
            parameters_count=1000000,
            memory_mb=5000.0,  # More than device_3 can handle (3072MB available)
            compute_requirement=5.0,
        )

        initial_plan = ShardingPlan(
            model_name=temp_model_dir,
            total_shards=1,
            shards=[problematic_shard],
            activation_routing={"problem_shard": []},
            memory_efficiency=0.8,
            compute_balance_score=0.7,
        )

        optimized_plan = await sharding_engine._optimize_sharding_plan(initial_plan, device_profiles)

        # Should have resolved memory constraints
        for shard in optimized_plan.shards:
            device = next(d for d in device_profiles if d.device_id == shard.device_id)
            assert shard.memory_mb <= device.available_memory_mb

    @pytest.mark.asyncio
    async def test_hybrid_sharding_strategy(self, sharding_engine, device_profiles, temp_model_dir):
        """Test hybrid sharding strategy"""
        mock_analysis = {
            "model_path": temp_model_dir,
            "num_layers": 8,
            "layer_memory_mb": 150.0,
            "layer_compute_score": 1.5,
            "total_memory_mb": 1200.0,
        }

        with patch.object(sharding_engine, "_analyze_model", return_value=mock_analysis):
            plan = await sharding_engine._create_hybrid_plan(mock_analysis, device_profiles)

            # Should choose the better of memory-aware or compute-balanced
            assert isinstance(plan, ShardingPlan)
            assert plan.total_shards > 0

            # Should have reasonable efficiency scores
            assert 0.0 <= plan.memory_efficiency <= 1.0
            assert 0.0 <= plan.compute_balance_score <= 1.0

    @pytest.mark.asyncio
    async def test_full_sharding_workflow(self, sharding_engine, device_profiles, temp_model_dir):
        """Test complete sharding workflow"""
        # Mock dependencies
        with (
            patch.object(sharding_engine, "_analyze_model") as mock_analyze,
            patch.object(sharding_engine, "_get_device_profiles", return_value=device_profiles),
        ):
            mock_analyze.return_value = {
                "model_path": temp_model_dir,
                "num_layers": 6,
                "layer_memory_mb": 200.0,
                "layer_compute_score": 2.0,
                "total_memory_mb": 1200.0,
                "can_split_attention": True,
                "optimal_shard_count": 3,
            }

            # Execute sharding
            plan = await sharding_engine.shard_model(temp_model_dir, strategy=ShardingStrategy.HYBRID)

            # Verify results
            assert sharding_engine.current_sharding_plan == plan
            assert len(sharding_engine.active_shards) == plan.total_shards
            assert len(sharding_engine.device_assignments) > 0
            assert sharding_engine.stats["models_sharded"] == 1

            # Verify P2P notification was sent
            sharding_engine.p2p_node.broadcast_to_peers.assert_called()

    @pytest.mark.asyncio
    async def test_device_failure_handling(self, sharding_engine, device_profiles, temp_model_dir):
        """Test handling of device failures during sharding"""
        # Create initial sharding plan
        with (
            patch.object(sharding_engine, "_analyze_model") as mock_analyze,
            patch.object(sharding_engine, "_get_device_profiles", return_value=device_profiles),
        ):
            mock_analyze.return_value = {
                "model_path": temp_model_dir,
                "num_layers": 3,
                "layer_memory_mb": 100.0,
                "layer_compute_score": 1.0,
                "total_memory_mb": 300.0,
            }

            plan = await sharding_engine.shard_model(temp_model_dir)

            # Simulate device failure by removing device from profiles
            failed_device_id = plan.shards[0].device_id
            remaining_devices = [d for d in device_profiles if d.device_id != failed_device_id]

            # Test resharding with failed device
            with patch.object(sharding_engine, "_get_device_profiles", return_value=remaining_devices):
                new_plan = await sharding_engine._create_memory_aware_plan(mock_analyze.return_value, remaining_devices)

                # Should not assign shards to failed device
                assert all(shard.device_id != failed_device_id for shard in new_plan.shards)

    @pytest.mark.asyncio
    async def test_memory_constraint_validation(self, sharding_engine, device_profiles):
        """Test memory constraint validation"""
        # Create shard that exceeds device memory
        oversized_shard = ModelShard(
            shard_id="oversized",
            device_id="device_3",  # 3072MB available
            layer_indices=[0, 1, 2],
            parameters_count=5000000,
            memory_mb=4000.0,  # Exceeds available memory
            compute_requirement=3.0,
        )

        plan = ShardingPlan(
            model_name="test_model",
            total_shards=1,
            shards=[oversized_shard],
            activation_routing={"oversized": []},
            memory_efficiency=0.5,
            compute_balance_score=0.5,
        )

        # Optimization should handle the constraint violation
        optimized_plan = await sharding_engine._optimize_sharding_plan(plan, device_profiles)

        # Should either split the shard or move it to a larger device
        for shard in optimized_plan.shards:
            device = next(d for d in device_profiles if d.device_id == shard.device_id)
            assert shard.memory_mb <= device.available_memory_mb

    @pytest.mark.asyncio
    async def test_activation_routing(self, sharding_engine, device_profiles, temp_model_dir):
        """Test activation routing between shards"""
        mock_analysis = {
            "model_path": temp_model_dir,
            "num_layers": 6,
            "layer_memory_mb": 100.0,
            "layer_compute_score": 1.0,
            "total_memory_mb": 600.0,
        }

        with patch.object(sharding_engine, "_analyze_model", return_value=mock_analysis):
            plan = await sharding_engine._create_sequential_plan(mock_analysis, device_profiles)

            # Verify activation routing forms a proper chain
            assert len(plan.activation_routing) == len(plan.shards)

            # First shard should have no dependencies
            first_shard = plan.shards[0]
            assert plan.activation_routing[first_shard.shard_id] == []

            # Other shards should depend on previous shard
            for i in range(1, len(plan.shards)):
                current_shard = plan.shards[i]
                previous_shard = plan.shards[i - 1]
                assert plan.activation_routing[current_shard.shard_id] == [previous_shard.shard_id]

    def test_memory_efficiency_calculation(self, sharding_engine, device_profiles):
        """Test memory efficiency calculation"""
        shards = [
            ModelShard("shard1", "device_1", [0, 1], 1000, 100.0, 1.0),
            ModelShard("shard2", "device_2", [2, 3], 1000, 200.0, 1.0),
            ModelShard("shard3", "device_3", [4, 5], 1000, 150.0, 1.0),
        ]

        efficiency = sharding_engine._calculate_memory_efficiency(shards, device_profiles)

        # Total used: 450MB, Total available: ~21GB (sum of device memory)
        total_available = sum(d.available_memory_mb for d in device_profiles)
        expected_efficiency = 450.0 / total_available

        assert abs(efficiency - expected_efficiency) < 0.01

    def test_compute_balance_calculation(self, sharding_engine, device_profiles):
        """Test compute balance score calculation"""
        shards = [
            ModelShard("shard1", "device_1", [0, 1], 1000, 100.0, 2.0),
            ModelShard("shard2", "device_2", [2, 3], 1000, 100.0, 1.0),
            ModelShard("shard3", "device_3", [4, 5], 1000, 100.0, 0.5),
        ]

        balance_score = sharding_engine._calculate_compute_balance(shards, device_profiles)

        # Should be between 0 and 1
        assert 0.0 <= balance_score <= 1.0

        # Better balance should have higher score
        balanced_shards = [
            ModelShard("shard1", "device_1", [0, 1], 1000, 100.0, 3.0),  # Matches device compute
            ModelShard("shard2", "device_2", [2, 3], 1000, 100.0, 2.0),  # Proportional
            ModelShard("shard3", "device_3", [4, 5], 1000, 100.0, 1.0),  # Proportional
        ]

        balanced_score = sharding_engine._calculate_compute_balance(balanced_shards, device_profiles)
        assert balanced_score >= balance_score  # More balanced should score higher

    @pytest.mark.asyncio
    async def test_cleanup_sharding(self, sharding_engine, temp_model_dir):
        """Test sharding cleanup"""
        # Create a sharding plan first
        mock_plan = ShardingPlan(
            model_name=temp_model_dir,
            total_shards=2,
            shards=[
                ModelShard("shard1", "device_1", [0, 1], 1000, 100.0, 1.0),
                ModelShard("shard2", "device_2", [2, 3], 1000, 100.0, 1.0),
            ],
            activation_routing={"shard1": [], "shard2": ["shard1"]},
            memory_efficiency=0.8,
            compute_balance_score=0.7,
        )

        sharding_engine.current_sharding_plan = mock_plan
        sharding_engine.active_shards = {s.shard_id: s for s in mock_plan.shards}
        sharding_engine.device_assignments = {
            "device_1": ["shard1"],
            "device_2": ["shard2"],
        }

        # Cleanup
        await sharding_engine.cleanup_sharding()

        # Verify cleanup
        assert sharding_engine.current_sharding_plan is None
        assert len(sharding_engine.active_shards) == 0
        assert len(sharding_engine.device_assignments) == 0

        # Verify P2P notification was sent
        sharding_engine.p2p_node.broadcast_to_peers.assert_called()

    def test_sharding_status(self, sharding_engine):
        """Test sharding status reporting"""
        status = sharding_engine.get_sharding_status()

        assert "active_plan" in status
        assert "plan_details" in status
        assert "statistics" in status
        assert "cache_size" in status

        assert status["active_plan"] is False  # No active plan initially
        assert status["statistics"]["models_sharded"] == 0
