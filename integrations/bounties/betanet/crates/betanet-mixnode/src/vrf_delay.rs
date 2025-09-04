//! VRF-based delay calculation

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::Result;

/// Calculate VRF-based delay
pub async fn calculate_vrf_delay(min_delay: &Duration, max_delay: &Duration) -> Result<Duration> {
    #[cfg(feature = "vrf")]
    {
        use rand::rngs::OsRng;
        use schnorrkel::{signing_context, Keypair};

        // Generate ephemeral keypair for delay calculation
        let mut csprng = OsRng;
        let keypair = Keypair::generate_with(&mut csprng);

        // Use current timestamp as the VRF message
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_be_bytes();
        let ctx = signing_context(b"betanet-mixnode-delay");

        // Generate VRF proof and extract randomness
        let (io, proof, _) = keypair.vrf_sign(ctx.bytes(&now));
        // Verify proof for safety
        keypair
            .public
            .vrf_verify(ctx.bytes(&now), &io.to_preout(), &proof)
            .map_err(|e| crate::MixnodeError::Vrf(format!("VRF proof verification failed: {e}")))?;

        // Map VRF output to delay range
        let bytes: [u8; 8] = io.make_bytes(b"delay");
        let value = u64::from_be_bytes(bytes);
        let range = max_delay.as_millis() - min_delay.as_millis();
        let delay_offset = (value as u128) % range;

        Ok(Duration::from_millis((min_delay.as_millis() + delay_offset) as u64))
    }

    #[cfg(not(feature = "vrf"))]
    {
        Err(crate::MixnodeError::Vrf(
            "VRF feature not enabled".to_string(),
        ))
    }
}

/// VRF key pair
#[cfg(feature = "vrf")]
pub struct VrfKeyPair {
    keypair: schnorrkel::Keypair,
}

#[cfg(feature = "vrf")]
impl VrfKeyPair {
    /// Generate new VRF keypair using system randomness
    pub fn generate() -> Self {
        use rand::rngs::OsRng;
        let mut csprng = OsRng;
        let keypair = schnorrkel::Keypair::generate_with(&mut csprng);
        Self { keypair }
    }

    /// Construct keypair from a fixed seed (useful for tests)
    pub fn from_seed(seed: [u8; 32]) -> Result<Self> {
        use schnorrkel::{ExpansionMode, MiniSecretKey};
        let mini = MiniSecretKey::from_bytes(&seed)
            .map_err(|e| crate::MixnodeError::Vrf(format!("Invalid seed: {e}")))?;
        let keypair = mini.expand_to_keypair(ExpansionMode::Ed25519);
        Ok(Self { keypair })
    }

    /// Get public key bytes
    pub fn public_key(&self) -> [u8; 32] {
        self.keypair.public.to_bytes()
    }

    /// Prove value and return VRF proof
    pub fn prove(&self, message: &[u8]) -> Result<VrfProof> {
        use schnorrkel::signing_context;
        let ctx = signing_context(b"betanet-mixnode-vrf");
        let (io, proof, _) = self.keypair.vrf_sign(ctx.bytes(message));
        Ok(VrfProof { io, proof })
    }

    /// Verify proof for message
    pub fn verify(&self, message: &[u8], proof: &VrfProof) -> bool {
        use schnorrkel::signing_context;
        let ctx = signing_context(b"betanet-mixnode-vrf");
        self.keypair
            .public
            .vrf_verify(ctx.bytes(message), &proof.io.to_preout(), &proof.proof)
            .is_ok()
    }
}

/// VRF proof
#[cfg(feature = "vrf")]
pub struct VrfProof {
    /// Input/output pair from VRF
    pub io: schnorrkel::vrf::VRFInOut,
    /// VRF proof bytes
    pub proof: schnorrkel::vrf::VRFProof,
}

#[cfg(feature = "vrf")]
impl VrfProof {
    /// Extract delay from VRF output
    pub fn extract_delay(&self, min_delay: Duration, max_delay: Duration) -> Duration {
        let bytes: [u8; 8] = self.io.make_bytes(b"delay");
        let value = u64::from_be_bytes(bytes);
        let range = max_delay.as_millis() - min_delay.as_millis();
        let delay_offset = (value as u128) % range;
        Duration::from_millis((min_delay.as_millis() + delay_offset) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vrf_delay_calculation() {
        let min_delay = Duration::from_millis(100);
        let max_delay = Duration::from_millis(1000);

        #[cfg(feature = "vrf")]
        {
            let delay = calculate_vrf_delay(&min_delay, &max_delay).await.unwrap();
            assert!(delay >= min_delay);
            assert!(delay <= max_delay);
        }

        #[cfg(not(feature = "vrf"))]
        {
            let result = calculate_vrf_delay(&min_delay, &max_delay).await;
            assert!(result.is_err());
        }
    }

    #[cfg(feature = "vrf")]
    #[test]
    fn test_vrf_keypair() {
        let keypair = VrfKeyPair::generate();
        let message = b"test message";

        let proof = keypair.prove(message).unwrap();
        assert!(keypair.verify(message, &proof));

        let min_delay = Duration::from_millis(100);
        let max_delay = Duration::from_millis(1000);
        let delay = proof.extract_delay(min_delay, max_delay);

        assert!(delay >= min_delay);
        assert!(delay <= max_delay);
    }
}
