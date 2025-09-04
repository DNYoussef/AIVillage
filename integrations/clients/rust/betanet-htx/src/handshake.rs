//! HTX handshake protocol

use bytes::Bytes;
use tracing::{debug, trace};

use crate::{HtxError, Result};

#[cfg(all(not(feature = "noise-xk"), not(feature = "insecure-handshake")))]
compile_error!(
    "Noise-XK negotiation is required. Enable the 'insecure-handshake' feature to bypass at your own risk."
);

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
    #[cfg(all(feature = "noise-xk", not(feature = "insecure-handshake")))]
    noise_handshake: Option<snow::HandshakeState>,
}

impl Handshake {
    /// Create a new handshake as initiator
    pub fn new_initiator() -> Result<Self> {
        debug!("Creating new initiator handshake");

        #[cfg(all(feature = "noise-xk", not(feature = "insecure-handshake")))]
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

        #[cfg(feature = "insecure-handshake")]
        Ok(Self {
            state: HandshakeState::Initial,
        })
    }

    /// Create a new handshake as responder
    pub fn new_responder() -> Result<Self> {
        debug!("Creating new responder handshake");

        #[cfg(all(feature = "noise-xk", not(feature = "insecure-handshake")))]
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

        #[cfg(feature = "insecure-handshake")]
        Ok(Self {
            state: HandshakeState::Initial,
        })
    }

    /// Process handshake message
    pub fn process_message(&mut self, message: &[u8]) -> Result<Option<Bytes>> {
        trace!("Processing handshake message of {} bytes", message.len());

        #[cfg(all(feature = "noise-xk", not(feature = "insecure-handshake")))]
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

        #[cfg(feature = "insecure-handshake")]
        {
            // Stub implementation when explicitly opted-out of Noise
            self.state = HandshakeState::Completed;
            return Ok(Some(Bytes::from(message.to_vec())));
        }

        // If we reach here with Noise disabled, something is wrong
        Err(HtxError::Handshake("Noise-XK handshake not negotiated".into()))
    }

    /// Generate next handshake message
    pub fn generate_message(&mut self) -> Result<Bytes> {
        trace!("Generating handshake message");

        #[cfg(all(feature = "noise-xk", not(feature = "insecure-handshake")))]
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

        #[cfg(feature = "insecure-handshake")]
        {
            // Stub implementation when explicitly opted-out of Noise
            self.state = HandshakeState::Completed;
            return Ok(Bytes::from("HELLO"));
        }

        Err(HtxError::Handshake("Noise-XK handshake not negotiated".into()))
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

    #[cfg(feature = "noise-xk")]
    #[test]
    fn handshake_fails_on_malformed_initiator_message() {
        let mut initiator = Handshake::new_initiator().unwrap();
        let mut responder = Handshake::new_responder().unwrap();

        // Initiator generates first message
        let mut msg = initiator.generate_message().unwrap().to_vec();
        // Corrupt the message
        if !msg.is_empty() {
            msg[0] ^= 0x01;
        }
        assert!(responder.process_message(&msg).is_err());
    }

    #[cfg(feature = "noise-xk")]
    #[test]
    fn handshake_fails_on_malformed_responder_message() {
        let mut initiator = Handshake::new_initiator().unwrap();
        let mut responder = Handshake::new_responder().unwrap();

        // Normal first flight
        let msg1 = initiator.generate_message().unwrap();
        responder.process_message(&msg1).unwrap();

        // Responder message gets corrupted
        let mut msg2 = responder.generate_message().unwrap().to_vec();
        if !msg2.is_empty() {
            msg2[0] ^= 0x01;
        }
        assert!(initiator.process_message(&msg2).is_err());
    }
}
