"""HypeRAG LoRA Adapter Registry.

Manages registration, validation, and retrieval of LoRA adapters.
Integrates with Guardian Gate for signature verification.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AdapterEntry:
    """Registry entry for a LoRA adapter."""

    adapter_id: str
    sha256: str
    domain: str
    base_model: str
    metrics: dict[str, float]
    training_config: dict[str, Any]
    created_at: str
    guardian_signature: str | None = None
    signed_at: str | None = None
    status: str = "pending"  # pending, approved, rejected, revoked

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterEntry":
        """Create from dictionary."""
        return cls(**data)


class LoRARegistry:
    """Registry for managing LoRA adapters."""

    def __init__(self, registry_path: Path, guardian_gate=None) -> None:
        self.registry_path = registry_path
        self.guardian_gate = guardian_gate
        self.entries: dict[str, AdapterEntry] = {}

        # Create registry file if it doesn't exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry()
        else:
            self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        try:
            with open(self.registry_path, encoding="utf-8") as f:
                data = json.load(f)
                self.entries = {k: AdapterEntry.from_dict(v) for k, v in data.get("adapters", {}).items()}
            logger.info(f"Loaded {len(self.entries)} adapters from registry")
        except Exception as e:
            logger.exception(f"Failed to load registry: {e}")
            self.entries = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "adapters": {k: v.to_dict() for k, v in self.entries.items()},
        }

        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved registry with {len(self.entries)} adapters")

    def register_adapter(
        self,
        adapter_path: Path,
        entry_data: dict[str, Any],
        require_guardian_approval: bool = True,
    ) -> str:
        """Register a new adapter."""
        # Create entry
        entry = AdapterEntry.from_dict(entry_data)

        # Verify adapter hash
        computed_hash = self._compute_adapter_hash(adapter_path)
        if computed_hash != entry.sha256:
            msg = f"Adapter hash mismatch: expected {entry.sha256}, got {computed_hash}"
            raise ValueError(msg)

        # Check for duplicates
        if entry.adapter_id in self.entries:
            msg = f"Adapter {entry.adapter_id} already registered"
            raise ValueError(msg)

        # Guardian approval if required
        if require_guardian_approval and self.guardian_gate:
            logger.info(f"Requesting Guardian approval for adapter {entry.adapter_id}")

            # Prepare validation request
            validation_request = {
                "adapter_id": entry.adapter_id,
                "domain": entry.domain,
                "metrics": entry.metrics,
                "base_model": entry.base_model,
                "training_config": entry.training_config,
            }

            # Get Guardian decision
            decision = self.guardian_gate.validate_adapter(validation_request)

            if decision["decision"] == "REJECT":
                entry.status = "rejected"
                logger.warning(f"Guardian rejected adapter {entry.adapter_id}: {decision['reason']}")
            elif decision["decision"] == "QUARANTINE":
                entry.status = "quarantine"
                logger.info(f"Guardian quarantined adapter {entry.adapter_id}: {decision['reason']}")
            else:  # APPLY
                entry.status = "approved"
                entry.guardian_signature = decision.get("signature")
                entry.signed_at = datetime.now(UTC).isoformat()
                logger.info(f"Guardian approved adapter {entry.adapter_id}")
        # Preserve original status if no Guardian (for testing)
        # Only set signed_at for approved adapters
        elif entry.status == "approved":
            entry.signed_at = datetime.now(UTC).isoformat()

        # Add to registry
        self.entries[entry.adapter_id] = entry
        self._save_registry()

        logger.info(f"Registered adapter {entry.adapter_id} with status: {entry.status}")
        return entry.adapter_id

    def get_adapter(self, adapter_id: str) -> AdapterEntry | None:
        """Get adapter by ID."""
        return self.entries.get(adapter_id)

    def list_adapters(self, domain: str | None = None, status: str | None = None) -> list[AdapterEntry]:
        """List adapters with optional filtering."""
        adapters = list(self.entries.values())

        if domain:
            adapters = [a for a in adapters if a.domain == domain]

        if status:
            adapters = [a for a in adapters if a.status == status]

        # Sort by creation date (newest first)
        adapters.sort(key=lambda a: a.created_at, reverse=True)

        return adapters

    def verify_adapter(self, adapter_id: str, adapter_path: Path) -> bool:
        """Verify adapter integrity."""
        entry = self.get_adapter(adapter_id)
        if not entry:
            logger.error(f"Adapter {adapter_id} not found in registry")
            return False

        # Compute current hash
        current_hash = self._compute_adapter_hash(adapter_path)

        # Verify hash matches
        if current_hash != entry.sha256:
            logger.error(f"Adapter {adapter_id} hash mismatch: expected {entry.sha256}, got {current_hash}")
            return False

        # Verify Guardian signature if present
        if entry.guardian_signature and self.guardian_gate:
            signature_valid = self.guardian_gate.verify_signature(
                data=entry.to_dict(), signature=entry.guardian_signature
            )
            if not signature_valid:
                logger.error(f"Invalid Guardian signature for adapter {adapter_id}")
                return False

        logger.info(f"Adapter {adapter_id} verification successful")
        return True

    def revoke_adapter(self, adapter_id: str, reason: str) -> bool:
        """Revoke an adapter."""
        entry = self.get_adapter(adapter_id)
        if not entry:
            return False

        entry.status = "revoked"
        self._save_registry()

        logger.warning(f"Revoked adapter {adapter_id}: {reason}")
        return True

    def get_best_adapter(self, domain: str, metric: str = "accuracy") -> AdapterEntry | None:
        """Get the best performing approved adapter for a domain."""
        domain_adapters = [a for a in self.entries.values() if a.domain == domain and a.status == "approved"]

        if not domain_adapters:
            return None

        # Sort by metric (higher is better)
        domain_adapters.sort(key=lambda a: a.metrics.get(metric, 0), reverse=True)

        return domain_adapters[0]

    def _compute_adapter_hash(self, adapter_path: Path) -> str:
        """Compute SHA256 hash of adapter files."""
        sha256_hash = hashlib.sha256()

        # Hash all weight files
        weight_files = list(adapter_path.glob("*.bin")) + list(adapter_path.glob("*.safetensors"))

        for weight_file in sorted(weight_files):
            with open(weight_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def export_registry(self, export_path: Path, format: str = "yaml") -> None:
        """Export registry in different formats."""
        data = {
            "version": "1.0",
            "exported_at": datetime.now(UTC).isoformat(),
            "total_adapters": len(self.entries),
            "adapters_by_domain": {},
            "adapters_by_status": {},
            "adapters": [],
        }

        # Aggregate statistics
        for entry in self.entries.values():
            # By domain
            if entry.domain not in data["adapters_by_domain"]:
                data["adapters_by_domain"][entry.domain] = 0
            data["adapters_by_domain"][entry.domain] += 1

            # By status
            if entry.status not in data["adapters_by_status"]:
                data["adapters_by_status"][entry.status] = 0
            data["adapters_by_status"][entry.status] += 1

            # Add adapter summary
            data["adapters"].append(
                {
                    "id": entry.adapter_id,
                    "domain": entry.domain,
                    "status": entry.status,
                    "metrics": entry.metrics,
                    "created_at": entry.created_at,
                }
            )

        # Export based on format
        if format == "yaml":
            with open(export_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:  # JSON
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        logger.info(f"Exported registry to {export_path} in {format} format")
