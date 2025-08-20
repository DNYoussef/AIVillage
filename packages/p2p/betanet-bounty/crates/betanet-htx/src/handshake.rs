//! HTX handshake protocol

use bytes::Bytes;
use tracing::{debug, trace};

use crate::{HtxError, Result};

/// Handshake state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    /// Initial state
    Initial,
    /// Sent hello
    SentHello,
    /// Received hello
    ReceivedHello,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// HTX handshake
pub struct Handshake {
    state: HandshakeState,
    #[cfg(feature = "noise-xk")]
    noise_handshake: Option<snow::HandshakeState>,
}

impl Handshake {
    /// Create a new handshake as initiator
    pub fn new_initiator() -> Result<Self> {
        debug!("Creating new initiator handshake");

        #[cfg(feature = "noise-xk")]
        {
            let params = "Noise_XK_25519_ChaChaPoly_SHA256";
            let builder = snow::Builder::new(params.parse().map_err(|e| {
                HtxError::Handshake(format!("Failed to parse Noise params: {}", e))
            })?);

            let keypair = builder.generate_keypair().map_err(|e| {
                HtxError::Handshake(format!("Failed to generate keypair: {}", e))
            })?;

            let noise_handshake = builder
                .local_private_key(&keypair.private)
                .build_initiator()
                .map_err(|e| HtxError::Handshake(format!("Failed to build initiator: {}", e)))?;

            Ok(Self {
                state: HandshakeState::Initial,
                noise_handshake: Some(noise_handshake),
            })
        }

        #[cfg(not(feature = "noise-xk"))]
        Ok(Self {
            state: HandshakeState::Initial,
        })
    }

    /// Create a new handshake as responder
    pub fn new_responder() -> Result<Self> {
        debug!("Creating new responder handshake");

        #[cfg(feature = "noise-xk")]
        {
            let params = "Noise_XK_25519_ChaChaPoly_SHA256";
            let builder = snow::Builder::new(params.parse().map_err(|e| {
                HtxError::Handshake(format!("Failed to parse Noise params: {}", e))
            })?);

            let keypair = builder.generate_keypair().map_err(|e| {
                HtxError::Handshake(format!("Failed to generate keypair: {}", e))
            })?;

            let noise_handshake = builder
                .local_private_key(&keypair.private)
                .build_responder()
                .map_err(|e| HtxError::Handshake(format!("Failed to build responder: {}", e)))?;

            Ok(Self {
                state: HandshakeState::Initial,
                noise_handshake: Some(noise_handshake),
            })
        }

        #[cfg(not(feature = "noise-xk"))]
        Ok(Self {
            state: HandshakeState::Initial,
        })
    }

    /// Process handshake message
    pub fn process_message(&mut self, message: &[u8]) -> Result<Option<Bytes>> {
        trace!("Processing handshake message of {} bytes", message.len());

        #[cfg(feature = "noise-xk")]
        {
            if let Some(ref mut noise) = self.noise_handshake {
                let mut buf = vec![0u8; 65535];
                let len = noise
                    .read_message(message, &mut buf)
                    .map_err(|e| HtxError::Handshake(format!("Failed to read message: {}", e)))?;
                buf.truncate(len);

                self.update_state();

                if noise.is_handshake_finished() {
                    self.state = HandshakeState::Completed;
                    debug!("Handshake completed");
                }

                return Ok(Some(Bytes::from(buf)));
            }
        }

        // Stub implementation without Noise
        self.state = HandshakeState::Completed;
        Ok(Some(Bytes::from(message.to_vec())))
    }

    /// Generate next handshake message
    pub fn generate_message(&mut self) -> Result<Bytes> {
        trace!("Generating handshake message");

        #[cfg(feature = "noise-xk")]
        {
            if let Some(ref mut noise) = self.noise_handshake {
                let mut buf = vec![0u8; 65535];
                let len = noise
                    .write_message(&[], &mut buf)
                    .map_err(|e| HtxError::Handshake(format!("Failed to write message: {}", e)))?;
                buf.truncate(len);

                self.update_state();

                return Ok(Bytes::from(buf));
            }
        }

        // Stub implementation without Noise
        Ok(Bytes::from("HELLO"))
    }

    /// Check if handshake is complete
    pub fn is_complete(&self) -> bool {
        self.state == HandshakeState::Completed
    }

    /// Update handshake state
    fn update_state(&mut self) {
        self.state = match self.state {
            HandshakeState::Initial => HandshakeState::SentHello,
            HandshakeState::SentHello => HandshakeState::ReceivedHello,
            HandshakeState::ReceivedHello => HandshakeState::Completed,
            other => other,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handshake_states() {
        let handshake = Handshake::new_initiator().unwrap();
        assert_eq!(handshake.state, HandshakeState::Initial);
        assert!(!handshake.is_complete());
    }
}
