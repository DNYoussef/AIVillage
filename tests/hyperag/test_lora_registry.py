"""
Tests for HypeRAG LoRA Registry
"""

# Import the registry module
import sys
import tempfile
from pathlib import Path

import pytest

from mcp_servers.hyperag.lora.registry import AdapterEntry, LoRARegistry

sys.path.append(str(Path(__file__).parent.parent.parent))


class MockGuardianGate:
    """Mock Guardian Gate for testing"""

    def validate_adapter(self, adapter_info):
        """Mock adapter validation"""
        metrics = adapter_info.get("metrics", {})
        accuracy = metrics.get("accuracy", 0)

        if accuracy >= 0.8:
            return {
                "decision": "APPLY",
                "confidence": 0.95,
                "reason": "Adapter meets quality standards",
                "signature": "guardian_v1:mock_signature_12345",
            }
        if accuracy >= 0.6:
            return {
                "decision": "QUARANTINE",
                "confidence": 0.6,
                "reason": "Metrics close to minimum thresholds",
            }
        return {
            "decision": "REJECT",
            "confidence": 0.9,
            "reason": "Metrics below threshold",
        }

    def verify_signature(self, data, signature):
        """Mock signature verification"""
        return signature == "guardian_v1:mock_signature_12345"


@pytest.fixture
def temp_registry():
    """Create a temporary registry file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        registry_path = Path(f.name)

    yield registry_path

    # Cleanup
    if registry_path.exists():
        registry_path.unlink()


@pytest.fixture
def mock_adapter_path(tmp_path):
    """Create a mock adapter directory with weight files"""
    adapter_dir = tmp_path / "test_adapter"
    adapter_dir.mkdir()

    # Create mock weight files
    weight_file = adapter_dir / "adapter_model.bin"
    weight_file.write_bytes(b"mock_weights_data_12345")

    return adapter_dir


@pytest.fixture
def sample_adapter_entry():
    """Sample adapter entry data"""
    return {
        "adapter_id": "medical_lora_20240115_120000",
        "sha256": "abc123def456",
        "domain": "medical",
        "base_model": "microsoft/phi-2",
        "metrics": {"accuracy": 0.85, "perplexity": 45.2, "eval_loss": 0.234},
        "training_config": {
            "peft_type": "lora",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
        },
        "created_at": "2024-01-15T12:00:00",
        "status": "pending",
    }


class TestAdapterEntry:
    """Test AdapterEntry dataclass"""

    def test_adapter_entry_creation(self, sample_adapter_entry):
        """Test creating adapter entry from dict"""
        entry = AdapterEntry.from_dict(sample_adapter_entry)

        assert entry.adapter_id == "medical_lora_20240115_120000"
        assert entry.domain == "medical"
        assert entry.metrics["accuracy"] == 0.85
        assert entry.status == "pending"
        assert entry.guardian_signature is None

    def test_adapter_entry_to_dict(self, sample_adapter_entry):
        """Test converting adapter entry to dict"""
        entry = AdapterEntry.from_dict(sample_adapter_entry)
        entry_dict = entry.to_dict()

        assert entry_dict["adapter_id"] == sample_adapter_entry["adapter_id"]
        assert entry_dict["metrics"] == sample_adapter_entry["metrics"]


class TestLoRARegistry:
    """Test LoRA Registry functionality"""

    def test_registry_initialization(self, temp_registry):
        """Test registry initialization"""
        registry = LoRARegistry(temp_registry)

        assert len(registry.entries) == 0
        assert temp_registry.exists()

    def test_register_adapter_success(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test successful adapter registration"""
        # Create registry with mock Guardian
        guardian = MockGuardianGate()
        registry = LoRARegistry(temp_registry, guardian_gate=guardian)

        # Update entry with correct hash
        sample_adapter_entry["sha256"] = registry._compute_adapter_hash(
            mock_adapter_path
        )
        sample_adapter_entry["metrics"]["accuracy"] = 0.85  # High accuracy for approval

        # Register adapter
        adapter_id = registry.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=True,
        )

        # Check registration
        assert adapter_id == sample_adapter_entry["adapter_id"]
        entry = registry.get_adapter(adapter_id)
        assert entry is not None
        assert entry.status == "approved"
        assert entry.guardian_signature == "guardian_v1:mock_signature_12345"

    def test_register_adapter_rejection(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test adapter rejection by Guardian"""
        guardian = MockGuardianGate()
        registry = LoRARegistry(temp_registry, guardian_gate=guardian)

        # Low accuracy for rejection
        sample_adapter_entry["sha256"] = registry._compute_adapter_hash(
            mock_adapter_path
        )
        sample_adapter_entry["metrics"]["accuracy"] = 0.5

        adapter_id = registry.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=True,
        )

        entry = registry.get_adapter(adapter_id)
        assert entry.status == "rejected"
        assert entry.guardian_signature is None

    def test_register_adapter_quarantine(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test adapter quarantine by Guardian"""
        guardian = MockGuardianGate()
        registry = LoRARegistry(temp_registry, guardian_gate=guardian)

        # Medium accuracy for quarantine
        sample_adapter_entry["sha256"] = registry._compute_adapter_hash(
            mock_adapter_path
        )
        sample_adapter_entry["metrics"]["accuracy"] = 0.65

        adapter_id = registry.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=True,
        )

        entry = registry.get_adapter(adapter_id)
        assert entry.status == "quarantine"

    def test_hash_verification_failure(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test adapter registration with wrong hash"""
        registry = LoRARegistry(temp_registry)

        # Wrong hash
        sample_adapter_entry["sha256"] = "wrong_hash_12345"

        with pytest.raises(ValueError, match="Adapter hash mismatch"):
            registry.register_adapter(
                adapter_path=mock_adapter_path,
                entry_data=sample_adapter_entry,
                require_guardian_approval=False,
            )

    def test_duplicate_adapter_registration(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test preventing duplicate adapter registration"""
        registry = LoRARegistry(temp_registry)

        sample_adapter_entry["sha256"] = registry._compute_adapter_hash(
            mock_adapter_path
        )

        # First registration
        registry.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=False,
        )

        # Second registration should fail
        with pytest.raises(ValueError, match="already registered"):
            registry.register_adapter(
                adapter_path=mock_adapter_path,
                entry_data=sample_adapter_entry,
                require_guardian_approval=False,
            )

    def test_list_adapters_filtering(self, temp_registry, mock_adapter_path):
        """Test listing adapters with filters"""
        registry = LoRARegistry(temp_registry)

        # Register multiple adapters
        adapters_data = [
            {"adapter_id": "medical_1", "domain": "medical", "status": "approved"},
            {"adapter_id": "medical_2", "domain": "medical", "status": "quarantine"},
            {"adapter_id": "movies_1", "domain": "movies", "status": "approved"},
            {"adapter_id": "finance_1", "domain": "finance", "status": "rejected"},
        ]

        for adapter_data in adapters_data:
            entry_data = {
                **adapter_data,
                "sha256": registry._compute_adapter_hash(mock_adapter_path),
                "base_model": "test",
                "metrics": {},
                "training_config": {},
                "created_at": "2024-01-15T12:00:00",
            }
            registry.register_adapter(
                adapter_path=mock_adapter_path,
                entry_data=entry_data,
                require_guardian_approval=False,
            )

        # Test domain filtering
        medical_adapters = registry.list_adapters(domain="medical")
        assert len(medical_adapters) == 2

        # Test status filtering
        approved_adapters = registry.list_adapters(status="approved")
        assert len(approved_adapters) == 2

        # Test combined filtering
        approved_medical = registry.list_adapters(domain="medical", status="approved")
        assert len(approved_medical) == 1
        assert approved_medical[0].adapter_id == "medical_1"

    def test_verify_adapter_integrity(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test adapter integrity verification"""
        guardian = MockGuardianGate()
        registry = LoRARegistry(temp_registry, guardian_gate=guardian)

        # Register adapter
        sample_adapter_entry["sha256"] = registry._compute_adapter_hash(
            mock_adapter_path
        )
        sample_adapter_entry["metrics"]["accuracy"] = 0.85

        adapter_id = registry.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=True,
        )

        # Verify integrity
        assert registry.verify_adapter(adapter_id, mock_adapter_path) is True

        # Verify with wrong adapter path
        wrong_path = mock_adapter_path.parent / "wrong_adapter"
        wrong_path.mkdir()
        (wrong_path / "adapter_model.bin").write_bytes(b"different_data")

        assert registry.verify_adapter(adapter_id, wrong_path) is False

    def test_revoke_adapter(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test adapter revocation"""
        registry = LoRARegistry(temp_registry)

        # Register adapter
        sample_adapter_entry["sha256"] = registry._compute_adapter_hash(
            mock_adapter_path
        )
        adapter_id = registry.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=False,
        )

        # Revoke adapter
        success = registry.revoke_adapter(adapter_id, "Security vulnerability found")
        assert success is True

        # Check status
        entry = registry.get_adapter(adapter_id)
        assert entry.status == "revoked"

    def test_get_best_adapter(self, temp_registry, mock_adapter_path):
        """Test getting best adapter for domain"""
        registry = LoRARegistry(temp_registry)

        # Register multiple medical adapters with different accuracies
        accuracies = [0.75, 0.85, 0.80]
        for i, accuracy in enumerate(accuracies):
            entry_data = {
                "adapter_id": f"medical_{i}",
                "sha256": registry._compute_adapter_hash(mock_adapter_path),
                "domain": "medical",
                "base_model": "test",
                "metrics": {"accuracy": accuracy},
                "training_config": {},
                "created_at": f"2024-01-15T12:0{i}:00",
                "status": "approved",
            }
            registry.register_adapter(
                adapter_path=mock_adapter_path,
                entry_data=entry_data,
                require_guardian_approval=False,
            )

        # Get best adapter
        best = registry.get_best_adapter("medical", metric="accuracy")
        assert best is not None
        assert best.adapter_id == "medical_1"
        assert best.metrics["accuracy"] == 0.85

    def test_export_registry(self, temp_registry, mock_adapter_path):
        """Test registry export functionality"""
        registry = LoRARegistry(temp_registry)

        # Register some adapters
        domains = ["medical", "medical", "movies", "finance"]
        statuses = ["approved", "quarantine", "approved", "rejected"]

        for i, (domain, status) in enumerate(zip(domains, statuses, strict=False)):
            entry_data = {
                "adapter_id": f"{domain}_{i}",
                "sha256": registry._compute_adapter_hash(mock_adapter_path),
                "domain": domain,
                "base_model": "test",
                "metrics": {"accuracy": 0.8},
                "training_config": {},
                "created_at": "2024-01-15T12:00:00",
                "status": status,
            }
            registry.register_adapter(
                adapter_path=mock_adapter_path,
                entry_data=entry_data,
                require_guardian_approval=False,
            )

        # Export to YAML
        export_path = temp_registry.parent / "export.yaml"
        registry.export_registry(export_path, format="yaml")

        assert export_path.exists()

        # Verify export content
        import yaml

        with open(export_path) as f:
            exported = yaml.safe_load(f)

        assert exported["total_adapters"] == 4
        assert exported["adapters_by_domain"]["medical"] == 2
        assert exported["adapters_by_status"]["approved"] == 2

    def test_registry_persistence(
        self, temp_registry, mock_adapter_path, sample_adapter_entry
    ):
        """Test registry persistence across instances"""
        # First instance - register adapter
        registry1 = LoRARegistry(temp_registry)
        sample_adapter_entry["sha256"] = registry1._compute_adapter_hash(
            mock_adapter_path
        )
        adapter_id = registry1.register_adapter(
            adapter_path=mock_adapter_path,
            entry_data=sample_adapter_entry,
            require_guardian_approval=False,
        )

        # Second instance - should load existing data
        registry2 = LoRARegistry(temp_registry)

        assert len(registry2.entries) == 1
        entry = registry2.get_adapter(adapter_id)
        assert entry is not None
        assert entry.domain == "medical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
