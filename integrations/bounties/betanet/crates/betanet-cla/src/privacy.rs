use bytes::{Buf, Bytes, BytesMut};
use thiserror::Error;

/// Privacy modes supported by the CLA
#[derive(Debug, Clone, Copy)]
pub enum PrivacyMode {
    /// No additional privacy protection
    Standard,
    /// Apply Sphinx wrapping to bundle bytes
    Strict,
}

#[derive(Debug, Error)]
pub enum PrivacyError {
    #[error("sphinx wrapping failed: {0}")]
    Sphinx(String),
}

/// Apply privacy transformation to encoded bundle bytes
pub fn apply_privacy(data: Bytes, mode: PrivacyMode) -> Result<Bytes, PrivacyError> {
    match mode {
        PrivacyMode::Standard => Ok(data),
        PrivacyMode::Strict => sphinx_wrap(data),
    }
}

/// Remove privacy layer from received bytes
pub fn remove_privacy(data: Bytes, mode: PrivacyMode) -> Result<Bytes, PrivacyError> {
    match mode {
        PrivacyMode::Standard => Ok(data),
        PrivacyMode::Strict => sphinx_unwrap(data),
    }
}

fn sphinx_wrap(data: Bytes) -> Result<Bytes, PrivacyError> {
    // Simplified Sphinx packet: magic header + length + payload
    let mut buf = BytesMut::with_capacity(8 + data.len());
    buf.extend_from_slice(b"SPHNX");
    buf.extend_from_slice(&(data.len() as u16).to_be_bytes());
    buf.extend_from_slice(&data);
    Ok(buf.freeze())
}

fn sphinx_unwrap(mut data: Bytes) -> Result<Bytes, PrivacyError> {
    if data.len() < 7 || !data.starts_with(b"SPHNX") {
        return Err(PrivacyError::Sphinx("invalid Sphinx packet".into()));
    }
    let _magic = data.split_to(5); // remove header
    let len = u16::from_be_bytes([data[0], data[1]]) as usize;
    data.advance(2);
    if data.len() < len {
        return Err(PrivacyError::Sphinx("truncated Sphinx payload".into()));
    }
    Ok(data.split_to(len))
}
