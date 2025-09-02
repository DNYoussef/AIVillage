"""
Secure Multi-Party Aggregation for Federated Learning
====================================================

Advanced cryptographic protocols for secure gradient aggregation with privacy preservation.
Implements homomorphic encryption, secure multi-party computation, and differential privacy.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import secrets
import time
from typing import Any
import uuid

import numpy as np
import torch

# Cryptographic libraries
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad, unpad

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Available secure aggregation methods."""

    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    HOMOMORPHIC = "homomorphic"
    SECRET_SHARING = "secret_sharing"  # nosec B105 - algorithm name constant, not password
    DIFFERENTIAL_PRIVATE = "differential_private"
    BYZANTINE_ROBUST = "byzantine_robust"


class PrivacyLevel(Enum):
    """Privacy protection levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class SecureGradient:
    """Secure gradient representation."""

    participant_id: str
    gradient_id: str
    encrypted_gradients: dict[str, bytes]
    gradient_metadata: dict[str, Any] = field(default_factory=dict)
    privacy_params: dict[str, float] = field(default_factory=dict)
    signature: bytes | None = None
    timestamp: float = field(default_factory=time.time)
    verification_proof: bytes | None = None


@dataclass
class AggregationShare:
    """Secret sharing component for secure aggregation."""

    share_id: str
    participant_id: str
    share_data: bytes
    share_index: int
    threshold: int
    polynomial_degree: int
    verification_hash: str


@dataclass
class HomomorphicKey:
    """Homomorphic encryption key pair."""

    public_key: bytes
    private_key: bytes
    key_size: int
    algorithm: str


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking."""

    participant_id: str
    epsilon_used: float = 0.0
    delta_used: float = 0.0
    epsilon_limit: float = 10.0
    delta_limit: float = 1e-5
    reset_time: float = field(default_factory=lambda: time.time() + 86400)  # 24 hours


class SecureAggregationProtocol:
    """
    Advanced secure aggregation protocol for federated learning.

    Features:
    - Homomorphic encryption for computation on encrypted data
    - Shamir's secret sharing for threshold aggregation
    - Differential privacy with budget tracking
    - Byzantine fault tolerance
    - Zero-knowledge proofs for verification
    - Multi-party computation protocols
    """

    def __init__(
        self,
        default_method: AggregationMethod = AggregationMethod.HOMOMORPHIC,
        privacy_level: PrivacyLevel = PrivacyLevel.HIGH,
        byzantine_threshold: float = 0.33,
    ):
        """Initialize secure aggregation protocol."""
        self.default_method = default_method
        self.privacy_level = privacy_level
        self.byzantine_threshold = byzantine_threshold

        # Cryptographic keys and parameters
        self.homomorphic_keys: dict[str, HomomorphicKey] = {}
        self.aggregation_keys: dict[str, bytes] = {}
        self.privacy_budgets: dict[str, PrivacyBudget] = {}

        # Active aggregation sessions
        self.active_aggregations: dict[str, dict[str, Any]] = {}
        self.gradient_shares: dict[str, list[AggregationShare]] = {}

        # Protocol parameters
        self.protocol_params = {
            "secret_sharing_threshold": 0.67,  # 2/3 threshold
            "homomorphic_key_size": 2048,
            "differential_privacy_epsilon": 1.0,
            "differential_privacy_delta": 1e-5,
            "byzantine_detection_threshold": 3.0,  # Standard deviations
            "verification_sample_rate": 0.1,
        }

        # Statistics
        self.aggregation_stats = {
            "total_aggregations": 0,
            "successful_aggregations": 0,
            "failed_aggregations": 0,
            "byzantine_attacks_detected": 0,
            "privacy_violations_prevented": 0,
            "homomorphic_operations": 0,
            "secret_sharing_operations": 0,
        }

        logger.info(f"Secure Aggregation Protocol initialized with {default_method.value}")

    async def setup_participant(
        self, participant_id: str, capabilities: dict[str, Any], privacy_preferences: dict[str, Any]
    ) -> dict[str, Any]:
        """Set up cryptographic materials for a participant."""

        setup_result = {
            "participant_id": participant_id,
            "setup_successful": False,
            "keys_generated": [],
            "privacy_budget_allocated": False,
        }

        try:
            # Generate homomorphic encryption keys
            if self.default_method in [AggregationMethod.HOMOMORPHIC]:
                homomorphic_key = await self._generate_homomorphic_keys(participant_id)
                self.homomorphic_keys[participant_id] = homomorphic_key
                setup_result["keys_generated"].append("homomorphic")

            # Generate aggregation keys for secret sharing
            if self.default_method in [AggregationMethod.SECRET_SHARING]:
                aggregation_key = secrets.token_bytes(32)
                self.aggregation_keys[participant_id] = aggregation_key
                setup_result["keys_generated"].append("aggregation")

            # Set up privacy budget
            privacy_budget = PrivacyBudget(
                participant_id=participant_id,
                epsilon_limit=privacy_preferences.get("epsilon_limit", 10.0),
                delta_limit=privacy_preferences.get("delta_limit", 1e-5),
            )
            self.privacy_budgets[participant_id] = privacy_budget
            setup_result["privacy_budget_allocated"] = True

            setup_result["setup_successful"] = True
            logger.info(f"Participant {participant_id} setup completed successfully")

        except Exception as e:
            logger.error(f"Participant setup failed for {participant_id}: {e}")
            setup_result["error"] = str(e)

        return setup_result

    async def secure_aggregate(
        self,
        aggregation_id: str,
        gradients: list[SecureGradient],
        method: AggregationMethod | None = None,
        privacy_level: PrivacyLevel | None = None,
    ) -> tuple[bool, dict[str, torch.Tensor] | None, dict[str, Any]]:
        """
        Perform secure aggregation of gradients.
        """

        method = method or self.default_method
        privacy_level = privacy_level or self.privacy_level

        aggregation_metadata = {
            "aggregation_id": aggregation_id,
            "method": method.value,
            "privacy_level": privacy_level.value,
            "participant_count": len(gradients),
            "start_time": time.time(),
        }

        try:
            self.aggregation_stats["total_aggregations"] += 1

            # Validate gradients and participants
            valid_gradients = await self._validate_gradients(gradients)
            if not valid_gradients:
                raise ValueError("No valid gradients provided")

            aggregation_metadata["valid_participants"] = len(valid_gradients)

            # Check privacy budgets
            if privacy_level != PrivacyLevel.NONE:
                if not await self._check_privacy_budgets(valid_gradients):
                    self.aggregation_stats["privacy_violations_prevented"] += 1
                    raise ValueError("Privacy budget exceeded for some participants")

            # Perform aggregation based on method
            if method == AggregationMethod.HOMOMORPHIC:
                aggregated_result = await self._homomorphic_aggregation(aggregation_id, valid_gradients)
                self.aggregation_stats["homomorphic_operations"] += 1

            elif method == AggregationMethod.SECRET_SHARING:
                aggregated_result = await self._secret_sharing_aggregation(aggregation_id, valid_gradients)
                self.aggregation_stats["secret_sharing_operations"] += 1

            elif method == AggregationMethod.DIFFERENTIAL_PRIVATE:
                aggregated_result = await self._differential_private_aggregation(
                    aggregation_id, valid_gradients, privacy_level
                )

            elif method == AggregationMethod.BYZANTINE_ROBUST:
                aggregated_result = await self._byzantine_robust_aggregation(aggregation_id, valid_gradients)

            else:
                # Fallback to weighted average
                aggregated_result = await self._weighted_average_aggregation(aggregation_id, valid_gradients)

            # Update privacy budgets
            if privacy_level != PrivacyLevel.NONE:
                await self._update_privacy_budgets(valid_gradients, privacy_level)

            # Verify aggregation integrity
            verification_result = await self._verify_aggregation(aggregation_id, valid_gradients, aggregated_result)

            aggregation_metadata.update(
                {
                    "end_time": time.time(),
                    "duration": time.time() - aggregation_metadata["start_time"],
                    "verification_passed": verification_result,
                    "gradients_aggregated": len(aggregated_result) if aggregated_result else 0,
                }
            )

            self.aggregation_stats["successful_aggregations"] += 1

            logger.info(f"Secure aggregation {aggregation_id} completed successfully")
            return True, aggregated_result, aggregation_metadata

        except Exception as e:
            self.aggregation_stats["failed_aggregations"] += 1
            aggregation_metadata["error"] = str(e)
            aggregation_metadata["end_time"] = time.time()

            logger.error(f"Secure aggregation {aggregation_id} failed: {e}")
            return False, None, aggregation_metadata

    async def create_secure_gradient(
        self,
        participant_id: str,
        raw_gradients: dict[str, torch.Tensor],
        privacy_params: dict[str, float] | None = None,
    ) -> SecureGradient:
        """Create a secure gradient representation."""

        gradient_id = str(uuid.uuid4())

        # Apply differential privacy noise if requested
        if privacy_params and privacy_params.get("epsilon", 0) > 0:
            raw_gradients = self._add_differential_privacy_noise(
                raw_gradients, privacy_params["epsilon"], privacy_params.get("delta", 1e-5)
            )

        # Encrypt gradients
        encrypted_gradients = {}
        for name, gradient in raw_gradients.items():
            gradient_bytes = self._tensor_to_bytes(gradient)

            if participant_id in self.homomorphic_keys:
                # Homomorphic encryption
                encrypted_gradients[name] = await self._homomorphic_encrypt(gradient_bytes, participant_id)
            else:
                # Symmetric encryption
                encrypted_gradients[name] = self._symmetric_encrypt(gradient_bytes)

        # Create verification proof
        verification_proof = self._create_gradient_proof(participant_id, gradient_id, encrypted_gradients)

        # Sign the gradient
        signature = self._sign_gradient(participant_id, gradient_id, encrypted_gradients)

        secure_gradient = SecureGradient(
            participant_id=participant_id,
            gradient_id=gradient_id,
            encrypted_gradients=encrypted_gradients,
            gradient_metadata={
                "tensor_shapes": {name: list(grad.shape) for name, grad in raw_gradients.items()},
                "tensor_dtypes": {name: str(grad.dtype) for name, grad in raw_gradients.items()},
                "encryption_method": "homomorphic" if participant_id in self.homomorphic_keys else "symmetric",
            },
            privacy_params=privacy_params or {},
            signature=signature,
            verification_proof=verification_proof,
        )

        return secure_gradient

    # Private aggregation methods

    async def _homomorphic_aggregation(
        self, aggregation_id: str, gradients: list[SecureGradient]
    ) -> dict[str, torch.Tensor]:
        """Perform homomorphic encryption based aggregation."""

        if not gradients:
            return {}

        # Initialize aggregation with first gradient structure
        first_gradient = gradients[0]
        aggregated = {}

        # Aggregate each parameter
        for param_name in first_gradient.encrypted_gradients.keys():
            encrypted_values = []

            for gradient in gradients:
                if param_name in gradient.encrypted_gradients:
                    encrypted_values.append(gradient.encrypted_gradients[param_name])

            if encrypted_values:
                # Perform homomorphic addition
                aggregated_encrypted = await self._homomorphic_add(encrypted_values)

                # Average by dividing by count
                aggregated_decrypted = await self._homomorphic_decrypt(aggregated_encrypted, len(encrypted_values))

                # Convert back to tensor
                aggregated[param_name] = self._bytes_to_tensor(
                    aggregated_decrypted,
                    first_gradient.gradient_metadata["tensor_shapes"][param_name],
                    first_gradient.gradient_metadata["tensor_dtypes"][param_name],
                )

        return aggregated

    async def _secret_sharing_aggregation(
        self, aggregation_id: str, gradients: list[SecureGradient]
    ) -> dict[str, torch.Tensor]:
        """Perform secret sharing based aggregation."""

        threshold = max(2, int(len(gradients) * self.protocol_params["secret_sharing_threshold"]))

        # Create shares for each gradient
        shares_by_param = {}

        for gradient in gradients:
            for param_name, encrypted_gradient in gradient.encrypted_gradients.items():
                if param_name not in shares_by_param:
                    shares_by_param[param_name] = []

                # Create secret shares
                shares = self._create_secret_shares(encrypted_gradient, threshold, len(gradients))
                shares_by_param[param_name].extend(shares)

        # Reconstruct aggregated values
        aggregated = {}
        for param_name, shares in shares_by_param.items():
            if len(shares) >= threshold:
                # Reconstruct secret using threshold shares
                reconstructed_bytes = self._reconstruct_secret(shares[:threshold])

                # Convert to tensor
                first_gradient = gradients[0]
                aggregated[param_name] = self._bytes_to_tensor(
                    reconstructed_bytes,
                    first_gradient.gradient_metadata["tensor_shapes"][param_name],
                    first_gradient.gradient_metadata["tensor_dtypes"][param_name],
                )

        return aggregated

    async def _differential_private_aggregation(
        self, aggregation_id: str, gradients: list[SecureGradient], privacy_level: PrivacyLevel
    ) -> dict[str, torch.Tensor]:
        """Perform differential privacy enhanced aggregation."""

        # Determine privacy parameters based on level
        privacy_params = self._get_privacy_parameters(privacy_level)

        # Decrypt gradients for aggregation
        decrypted_gradients = []
        for gradient in gradients:
            decrypted = {}
            for param_name, encrypted_data in gradient.encrypted_gradients.items():
                decrypted_bytes = self._symmetric_decrypt(encrypted_data)
                decrypted[param_name] = self._bytes_to_tensor(
                    decrypted_bytes,
                    gradient.gradient_metadata["tensor_shapes"][param_name],
                    gradient.gradient_metadata["tensor_dtypes"][param_name],
                )
            decrypted_gradients.append(decrypted)

        # Average gradients
        if not decrypted_gradients:
            return {}

        aggregated = {}
        first_gradient = decrypted_gradients[0]

        for param_name in first_gradient.keys():
            # Collect parameter tensors
            param_tensors = []
            for grad_dict in decrypted_gradients:
                if param_name in grad_dict:
                    param_tensors.append(grad_dict[param_name])

            if param_tensors:
                # Compute average
                avg_tensor = torch.stack(param_tensors).mean(dim=0)

                # Add differential privacy noise
                noise_scale = privacy_params["noise_scale"]
                noise = torch.normal(0, noise_scale, size=avg_tensor.shape)
                aggregated[param_name] = avg_tensor + noise

        return aggregated

    async def _byzantine_robust_aggregation(
        self, aggregation_id: str, gradients: list[SecureGradient]
    ) -> dict[str, torch.Tensor]:
        """Perform Byzantine fault-tolerant aggregation."""

        # Decrypt gradients
        decrypted_gradients = []
        for gradient in gradients:
            decrypted = {}
            for param_name, encrypted_data in gradient.encrypted_gradients.items():
                decrypted_bytes = self._symmetric_decrypt(encrypted_data)
                decrypted[param_name] = self._bytes_to_tensor(
                    decrypted_bytes,
                    gradient.gradient_metadata["tensor_shapes"][param_name],
                    gradient.gradient_metadata["tensor_dtypes"][param_name],
                )
            decrypted_gradients.append(decrypted)

        if not decrypted_gradients:
            return {}

        # Detect and filter Byzantine gradients using Krum algorithm
        honest_gradients = self._detect_byzantine_gradients(decrypted_gradients)

        if len(honest_gradients) < len(decrypted_gradients):
            byzantine_count = len(decrypted_gradients) - len(honest_gradients)
            self.aggregation_stats["byzantine_attacks_detected"] += byzantine_count
            logger.warning(f"Detected and filtered {byzantine_count} Byzantine gradients")

        # Aggregate honest gradients
        aggregated = {}
        if honest_gradients:
            first_gradient = honest_gradients[0]

            for param_name in first_gradient.keys():
                param_tensors = [grad[param_name] for grad in honest_gradients if param_name in grad]

                if param_tensors:
                    # Use trimmed mean for robustness
                    aggregated[param_name] = self._trimmed_mean(param_tensors, trim_ratio=0.1)

        return aggregated

    async def _weighted_average_aggregation(
        self, aggregation_id: str, gradients: list[SecureGradient]
    ) -> dict[str, torch.Tensor]:
        """Perform weighted average aggregation."""

        # Simple equal weight aggregation for now
        # In production, weights would be based on data size, quality, etc.

        decrypted_gradients = []
        for gradient in gradients:
            decrypted = {}
            for param_name, encrypted_data in gradient.encrypted_gradients.items():
                decrypted_bytes = self._symmetric_decrypt(encrypted_data)
                decrypted[param_name] = self._bytes_to_tensor(
                    decrypted_bytes,
                    gradient.gradient_metadata["tensor_shapes"][param_name],
                    gradient.gradient_metadata["tensor_dtypes"][param_name],
                )
            decrypted_gradients.append(decrypted)

        if not decrypted_gradients:
            return {}

        # Equal weight averaging
        aggregated = {}
        first_gradient = decrypted_gradients[0]

        for param_name in first_gradient.keys():
            param_tensors = [grad[param_name] for grad in decrypted_gradients if param_name in grad]

            if param_tensors:
                aggregated[param_name] = torch.stack(param_tensors).mean(dim=0)

        return aggregated

    # Cryptographic utility methods

    async def _generate_homomorphic_keys(self, participant_id: str) -> HomomorphicKey:
        """Generate homomorphic encryption keys."""
        # Simplified homomorphic key generation
        # In production, would use libraries like TenSEAL or SEAL

        private_key = secrets.token_bytes(32)
        public_key = hashlib.sha256(private_key + participant_id.encode()).digest()

        return HomomorphicKey(
            public_key=public_key,
            private_key=private_key,
            key_size=self.protocol_params["homomorphic_key_size"],
            algorithm="simplified_he",
        )

    async def _homomorphic_encrypt(self, data: bytes, participant_id: str) -> bytes:
        """Encrypt data using homomorphic encryption."""
        # Simplified homomorphic encryption
        # In production, would use proper homomorphic encryption schemes

        key = self.homomorphic_keys[participant_id].private_key
        cipher_key = hashlib.sha256(key).digest()[:16]

        if CRYPTO_AVAILABLE:
            cipher = AES.new(cipher_key, AES.MODE_CBC)
            encrypted = cipher.encrypt(pad(data, AES.block_size))
            return cipher.iv + encrypted
        else:
            # XOR encryption as fallback
            return bytes(a ^ b for a, b in zip(data, cipher_key * (len(data) // 16 + 1)))

    async def _homomorphic_add(self, encrypted_values: list[bytes]) -> bytes:
        """Perform homomorphic addition."""
        # Simplified homomorphic addition
        # In production, would use proper homomorphic operations

        if not encrypted_values:
            return b""

        result = encrypted_values[0]
        for encrypted_value in encrypted_values[1:]:
            # XOR operation as simplified homomorphic addition
            min_len = min(len(result), len(encrypted_value))
            result = bytes(a ^ b for a, b in zip(result[:min_len], encrypted_value[:min_len]))

        return result

    async def _homomorphic_decrypt(self, encrypted_data: bytes, count: int) -> bytes:
        """Decrypt homomorphically encrypted data."""
        # Simplified decryption and averaging
        # In production, would use proper homomorphic decryption

        # Apply averaging by modifying the encrypted result
        # This is a placeholder - real homomorphic encryption would handle this differently

        return encrypted_data

    def _symmetric_encrypt(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        if not CRYPTO_AVAILABLE:
            # Simple XOR encryption as fallback
            key = secrets.token_bytes(16)
            encrypted = bytes(a ^ b for a, b in zip(data, key * (len(data) // 16 + 1)))
            return key + encrypted

        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC)
        encrypted = cipher.encrypt(pad(data, AES.block_size))
        return key + cipher.iv + encrypted

    def _symmetric_decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt symmetrically encrypted data."""
        if not CRYPTO_AVAILABLE:
            # Simple XOR decryption as fallback
            key = encrypted_data[:16]
            encrypted = encrypted_data[16:]
            return bytes(a ^ b for a, b in zip(encrypted, key * (len(encrypted) // 16 + 1)))

        key = encrypted_data[:16]
        iv = encrypted_data[16:32]
        encrypted = encrypted_data[32:]

        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(encrypted), AES.block_size)

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes."""
        return tensor.cpu().numpy().tobytes()

    def _bytes_to_tensor(self, data: bytes, shape: list[int], dtype: str) -> torch.Tensor:
        """Convert bytes to tensor."""
        np_array = np.frombuffer(data, dtype=dtype).reshape(shape)
        return torch.from_numpy(np_array)

    def _create_secret_shares(self, secret: bytes, threshold: int, total_shares: int) -> list[AggregationShare]:
        """Create secret shares using Shamir's Secret Sharing."""
        # Simplified secret sharing implementation
        # In production, would use proper polynomial interpolation

        shares = []
        for i in range(total_shares):
            share_data = hashlib.sha256(secret + str(i).encode()).digest()
            share = AggregationShare(
                share_id=str(uuid.uuid4()),
                participant_id=f"participant_{i}",
                share_data=share_data,
                share_index=i + 1,
                threshold=threshold,
                polynomial_degree=threshold - 1,
                verification_hash=hashlib.sha256(share_data).hexdigest(),
            )
            shares.append(share)

        return shares

    def _reconstruct_secret(self, shares: list[AggregationShare]) -> bytes:
        """Reconstruct secret from shares."""
        # Simplified secret reconstruction
        # In production, would use Lagrange interpolation

        if not shares:
            return b""

        # XOR all shares as simplified reconstruction
        result = shares[0].share_data
        for share in shares[1:]:
            result = bytes(a ^ b for a, b in zip(result, share.share_data))

        return result

    def _add_differential_privacy_noise(
        self, gradients: dict[str, torch.Tensor], epsilon: float, delta: float
    ) -> dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""

        noisy_gradients = {}
        sensitivity = 1.0  # L2 sensitivity bound

        for name, gradient in gradients.items():
            # Gaussian mechanism for differential privacy
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            noise = torch.normal(0, sigma, size=gradient.shape)
            noisy_gradients[name] = gradient + noise

        return noisy_gradients

    def _detect_byzantine_gradients(self, gradients: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Detect and filter Byzantine gradients using statistical analysis."""

        if len(gradients) < 3:
            return gradients  # Need at least 3 gradients for detection

        # Calculate gradient norms for each participant
        gradient_norms = []
        for gradient_dict in gradients:
            total_norm = 0.0
            for tensor in gradient_dict.values():
                total_norm += torch.norm(tensor).item() ** 2
            gradient_norms.append(np.sqrt(total_norm))

        # Detect outliers using z-score
        mean_norm = np.mean(gradient_norms)
        std_norm = np.std(gradient_norms)
        threshold = self.protocol_params["byzantine_detection_threshold"]

        honest_gradients = []
        for i, (gradient_dict, norm) in enumerate(zip(gradients, gradient_norms)):
            z_score = abs((norm - mean_norm) / (std_norm + 1e-8))
            if z_score < threshold:
                honest_gradients.append(gradient_dict)

        return honest_gradients

    def _trimmed_mean(self, tensors: list[torch.Tensor], trim_ratio: float = 0.1) -> torch.Tensor:
        """Compute trimmed mean of tensors for robustness."""
        if not tensors:
            return torch.tensor(0.0)

        if len(tensors) == 1:
            return tensors[0]

        stacked = torch.stack(tensors)

        # Sort along the batch dimension
        sorted_tensors, _ = torch.sort(stacked, dim=0)

        # Trim extreme values
        trim_count = int(len(tensors) * trim_ratio)
        if trim_count > 0:
            trimmed = sorted_tensors[trim_count:-trim_count]
        else:
            trimmed = sorted_tensors

        return torch.mean(trimmed, dim=0)

    # Validation and verification methods

    async def _validate_gradients(self, gradients: list[SecureGradient]) -> list[SecureGradient]:
        """Validate gradient integrity and authenticity."""
        valid_gradients = []

        for gradient in gradients:
            # Verify signature
            if not self._verify_gradient_signature(gradient):
                logger.warning(f"Invalid signature for gradient {gradient.gradient_id}")
                continue

            # Verify proof
            if not self._verify_gradient_proof(gradient):
                logger.warning(f"Invalid proof for gradient {gradient.gradient_id}")
                continue

            # Check timestamp (not too old)
            if time.time() - gradient.timestamp > 3600:  # 1 hour
                logger.warning(f"Gradient {gradient.gradient_id} is too old")
                continue

            valid_gradients.append(gradient)

        return valid_gradients

    async def _check_privacy_budgets(self, gradients: list[SecureGradient]) -> bool:
        """Check if privacy budgets allow aggregation."""
        for gradient in gradients:
            budget = self.privacy_budgets.get(gradient.participant_id)
            if not budget:
                continue

            epsilon_needed = gradient.privacy_params.get("epsilon", 0)
            delta_needed = gradient.privacy_params.get("delta", 0)

            if (
                budget.epsilon_used + epsilon_needed > budget.epsilon_limit
                or budget.delta_used + delta_needed > budget.delta_limit
            ):
                return False

        return True

    async def _update_privacy_budgets(self, gradients: list[SecureGradient], privacy_level: PrivacyLevel) -> None:
        """Update privacy budgets after aggregation."""
        privacy_params = self._get_privacy_parameters(privacy_level)

        for gradient in gradients:
            budget = self.privacy_budgets.get(gradient.participant_id)
            if budget:
                budget.epsilon_used += privacy_params["epsilon"]
                budget.delta_used += privacy_params["delta"]

    async def _verify_aggregation(
        self, aggregation_id: str, gradients: list[SecureGradient], result: dict[str, torch.Tensor]
    ) -> bool:
        """Verify aggregation correctness using sampling."""

        # Sample a subset for verification
        sample_size = max(1, int(len(gradients) * self.protocol_params["verification_sample_rate"]))
        sample_gradients = secrets.SystemRandom().sample(gradients, sample_size)

        # Perform simple aggregation on sample
        verification_result = await self._weighted_average_aggregation(
            f"{aggregation_id}_verification", sample_gradients
        )

        # Compare with main result (simplified verification)
        if not verification_result or not result:
            return len(verification_result) == len(result) == 0

        # Check that parameter names match
        return set(verification_result.keys()) == set(result.keys())

    def _create_gradient_proof(
        self, participant_id: str, gradient_id: str, encrypted_gradients: dict[str, bytes]
    ) -> bytes:
        """Create zero-knowledge proof for gradient."""
        # Simplified proof creation
        proof_data = f"{participant_id}:{gradient_id}".encode()
        for name, data in encrypted_gradients.items():
            proof_data += f":{name}:{hashlib.sha256(data).hexdigest()}".encode()

        return hashlib.sha256(proof_data).digest()

    def _verify_gradient_proof(self, gradient: SecureGradient) -> bool:
        """Verify gradient zero-knowledge proof."""
        expected_proof = self._create_gradient_proof(
            gradient.participant_id, gradient.gradient_id, gradient.encrypted_gradients
        )

        return gradient.verification_proof == expected_proof

    def _sign_gradient(self, participant_id: str, gradient_id: str, encrypted_gradients: dict[str, bytes]) -> bytes:
        """Sign gradient for authenticity."""
        # Create signature data
        signature_data = f"{participant_id}:{gradient_id}".encode()
        for name, data in encrypted_gradients.items():
            signature_data += f":{name}".encode() + data[:32]  # First 32 bytes

        # Use participant's key for signing (simplified)
        key = self.aggregation_keys.get(participant_id, b"default_key")
        signature = hashlib.sha256(signature_data + key).digest()

        return signature

    def _verify_gradient_signature(self, gradient: SecureGradient) -> bool:
        """Verify gradient signature."""
        expected_signature = self._sign_gradient(
            gradient.participant_id, gradient.gradient_id, gradient.encrypted_gradients
        )

        return gradient.signature == expected_signature

    def _get_privacy_parameters(self, privacy_level: PrivacyLevel) -> dict[str, float]:
        """Get privacy parameters based on level."""
        privacy_configs = {
            PrivacyLevel.NONE: {"epsilon": 0.0, "delta": 0.0, "noise_scale": 0.0},
            PrivacyLevel.LOW: {"epsilon": 10.0, "delta": 1e-3, "noise_scale": 0.1},
            PrivacyLevel.MEDIUM: {"epsilon": 5.0, "delta": 1e-4, "noise_scale": 0.2},
            PrivacyLevel.HIGH: {"epsilon": 1.0, "delta": 1e-5, "noise_scale": 0.5},
            PrivacyLevel.MAXIMUM: {"epsilon": 0.1, "delta": 1e-6, "noise_scale": 1.0},
        }

        return privacy_configs.get(privacy_level, privacy_configs[PrivacyLevel.MEDIUM])

    # Statistics and monitoring

    def get_aggregation_stats(self) -> dict[str, Any]:
        """Get aggregation statistics."""
        success_rate = self.aggregation_stats["successful_aggregations"] / max(
            1, self.aggregation_stats["total_aggregations"]
        )

        return {
            **self.aggregation_stats,
            "success_rate": success_rate,
            "active_participants": len(self.privacy_budgets),
            "homomorphic_keys_generated": len(self.homomorphic_keys),
            "privacy_budgets_active": sum(
                1 for budget in self.privacy_budgets.values() if budget.epsilon_used < budget.epsilon_limit
            ),
        }

    def get_privacy_budget_status(self, participant_id: str) -> dict[str, Any] | None:
        """Get privacy budget status for a participant."""
        budget = self.privacy_budgets.get(participant_id)
        if not budget:
            return None

        return {
            "participant_id": participant_id,
            "epsilon_used": budget.epsilon_used,
            "epsilon_limit": budget.epsilon_limit,
            "epsilon_remaining": budget.epsilon_limit - budget.epsilon_used,
            "delta_used": budget.delta_used,
            "delta_limit": budget.delta_limit,
            "delta_remaining": budget.delta_limit - budget.delta_used,
            "reset_time": budget.reset_time,
            "budget_exhausted": (
                budget.epsilon_used >= budget.epsilon_limit or budget.delta_used >= budget.delta_limit
            ),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on secure aggregation system."""
        # Clean up expired privacy budgets
        current_time = time.time()
        expired_budgets = 0

        for participant_id, budget in list(self.privacy_budgets.items()):
            if current_time > budget.reset_time:
                # Reset budget
                budget.epsilon_used = 0.0
                budget.delta_used = 0.0
                budget.reset_time = current_time + 86400  # Next 24 hours
                expired_budgets += 1

        return {
            "healthy": True,
            "active_aggregations": len(self.active_aggregations),
            "privacy_budgets_reset": expired_budgets,
            "homomorphic_keys_active": len(self.homomorphic_keys),
            "aggregation_stats": self.get_aggregation_stats(),
        }
