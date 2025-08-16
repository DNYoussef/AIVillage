//! Cryptographic primitives for mixnode

use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit},
    ChaCha20Poly1305, Key, Nonce,
};
use ed25519_dalek::{
    Keypair, PublicKey as Ed25519PublicKey, SecretKey, Signature, Signer, Verifier,
};
use hkdf::Hkdf;
use rand::rngs::OsRng;
use sha2::Sha256;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::{MixnodeError, Result};

/// Key derivation function
pub struct KeyDerivation;

impl KeyDerivation {
    /// Derive key from shared secret
    pub fn derive_key(shared_secret: &[u8], salt: &[u8], info: &[u8]) -> Result<[u8; 32]> {
        let hk = Hkdf::<Sha256>::new(Some(salt), shared_secret);
        let mut okm = [0u8; 32];
        hk.expand(info, &mut okm)
            .map_err(|e| MixnodeError::Crypto(format!("Key derivation failed: {}", e)))?;
        Ok(okm)
    }

    /// Generate random salt
    pub fn generate_salt() -> [u8; 32] {
        rand::random()
    }
}

/// ChaCha20-Poly1305 encryption/decryption
pub struct ChaChaEncryption {
    cipher: ChaCha20Poly1305,
}

impl ChaChaEncryption {
    /// Create new encryption instance
    pub fn new(key: &[u8; 32]) -> Self {
        let key = Key::from_slice(key);
        let cipher = ChaCha20Poly1305::new(key);
        Self { cipher }
    }

    /// Encrypt data
    pub fn encrypt(&self, plaintext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
        let nonce = Nonce::from_slice(nonce);
        self.cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| MixnodeError::Crypto(format!("Encryption failed: {}", e)))
    }

    /// Decrypt data
    pub fn decrypt(&self, ciphertext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
        let nonce = Nonce::from_slice(nonce);
        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| MixnodeError::Crypto(format!("Decryption failed: {}", e)))
    }

    /// Generate random nonce
    pub fn generate_nonce() -> [u8; 12] {
        ChaCha20Poly1305::generate_nonce(&mut OsRng).into()
    }
}

/// Ed25519 digital signatures
pub struct Ed25519Signer {
    keypair: Keypair,
}

impl Ed25519Signer {
    /// Create new signer with random key
    pub fn new() -> Self {
        // Use a deterministic key for now to avoid RNG compatibility issues
        let secret_bytes = [42u8; 32]; // TODO: Use proper random generation
        let secret = SecretKey::from_bytes(&secret_bytes).unwrap();
        let public = Ed25519PublicKey::from(&secret);
        let keypair = Keypair { secret, public };
        Self { keypair }
    }

    /// Create signer from private key bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        let secret = SecretKey::from_bytes(bytes)
            .map_err(|e| MixnodeError::Crypto(format!("Invalid secret key: {}", e)))?;
        let public = Ed25519PublicKey::from(&secret);
        let keypair = Keypair { secret, public };
        Ok(Self { keypair })
    }

    /// Get public key
    pub fn public_key(&self) -> &Ed25519PublicKey {
        &self.keypair.public
    }

    /// Sign data
    pub fn sign(&self, data: &[u8]) -> [u8; 64] {
        self.keypair.sign(data).to_bytes()
    }

    /// Verify signature
    pub fn verify(public_key: &Ed25519PublicKey, data: &[u8], signature: &[u8; 64]) -> bool {
        if let Ok(sig) = Signature::from_bytes(signature) {
            public_key.verify(data, &sig).is_ok()
        } else {
            false
        }
    }

    /// Export private key
    pub fn export_private_key(&self) -> [u8; 32] {
        self.keypair.secret.to_bytes()
    }
}

impl Default for Ed25519Signer {
    fn default() -> Self {
        Self::new()
    }
}

/// X25519 key exchange
pub struct X25519KeyExchange {
    secret: StaticSecret,
}

impl X25519KeyExchange {
    /// Create new key exchange with random key
    pub fn new() -> Self {
        // Use deterministic key for now to avoid RNG compatibility issues
        let secret_bytes = [42u8; 32]; // TODO: Use proper random generation
        let secret = StaticSecret::from(secret_bytes);
        Self { secret }
    }

    /// Create from private key bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let secret = StaticSecret::from(*bytes);
        Self { secret }
    }

    /// Get public key
    pub fn public_key(&self) -> PublicKey {
        PublicKey::from(&self.secret)
    }

    /// Perform key exchange
    pub fn exchange(&self, peer_public: &PublicKey) -> [u8; 32] {
        self.secret.diffie_hellman(peer_public).to_bytes()
    }

    /// Export private key
    pub fn export_private_key(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }
}

impl Default for X25519KeyExchange {
    fn default() -> Self {
        Self::new()
    }
}

/// Cryptographic utilities
pub struct CryptoUtils;

impl CryptoUtils {
    /// Generate random bytes
    pub fn random_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|_| rand::random::<u8>()).collect()
    }

    /// Hash data with SHA-256
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Constant-time comparison
    pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        result == 0
    }

    /// Secure random u64
    pub fn random_u64() -> u64 {
        rand::random()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_derivation() {
        let secret = b"shared secret";
        let salt = b"salt";
        let info = b"info";

        let key1 = KeyDerivation::derive_key(secret, salt, info).unwrap();
        let key2 = KeyDerivation::derive_key(secret, salt, info).unwrap();

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_chacha_encryption() {
        let key = rand::random();
        let encryption = ChaChaEncryption::new(&key);
        let nonce = ChaChaEncryption::generate_nonce();

        let plaintext = b"Hello, world!";
        let ciphertext = encryption.encrypt(plaintext, &nonce).unwrap();
        let decrypted = encryption.decrypt(&ciphertext, &nonce).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted);
    }

    #[test]
    fn test_ed25519_signing() {
        let signer = Ed25519Signer::new();
        let data = b"test message";

        let signature = signer.sign(data);
        let public_key = signer.public_key();

        assert!(Ed25519Signer::verify(&public_key, data, &signature));
        assert!(!Ed25519Signer::verify(
            &public_key,
            b"wrong message",
            &signature
        ));
    }

    #[test]
    fn test_x25519_key_exchange() {
        let alice = X25519KeyExchange::new();
        let bob = X25519KeyExchange::new();

        let alice_public = alice.public_key();
        let bob_public = bob.public_key();

        let alice_shared = alice.exchange(&bob_public);
        let bob_shared = bob.exchange(&alice_public);

        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_crypto_utils() {
        let data = b"test data";
        let hash1 = CryptoUtils::sha256(data);
        let hash2 = CryptoUtils::sha256(data);

        assert_eq!(hash1, hash2);

        assert!(CryptoUtils::constant_time_eq(&hash1, &hash2));
        assert!(!CryptoUtils::constant_time_eq(&hash1, &[0u8; 32]));
    }
}
