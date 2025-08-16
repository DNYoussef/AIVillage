//! HTX Frame Format Compliance Checks
//!
//! Validates HTX v1.1 frame format implementation for proper:
//! - Frame structure (uint24 length + varint stream_id + uint8 type + payload)
//! - Frame type definitions and validation
//! - Size limits and boundary checking
//! - Varint encoding/decoding compliance
//! - Buffer management and parsing logic
//! - Zero-copy optimization practices

use crate::{LintIssue, SeverityLevel, Result};
use crate::checks::{CheckRule, CheckContext};
use regex::Regex;
use async_trait::async_trait;

/// HTX frame structure compliance
pub struct FrameStructureRule;

#[async_trait]
impl CheckRule for FrameStructureRule {
    fn name(&self) -> &str {
        "frame-structure"
    }

    fn description(&self) -> &str {
        "Check HTX frame structure compliance with v1.1 specification"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check frame-related files
        if !context.file_path.to_string_lossy().contains("frame") &&
           !context.content.contains("Frame") &&
           !context.content.contains("HTX") {
            return Ok(issues);
        }

        // Check for MAX_FRAME_SIZE constant
        if context.content.contains("MAX_FRAME_SIZE") {
            let max_size_regex = Regex::new(r"MAX_FRAME_SIZE\s*:\s*usize\s*=\s*(\d+)").unwrap();
            for (line_num, line) in context.content.lines().enumerate() {
                if let Some(captures) = max_size_regex.captures(line) {
                    if let Some(size_str) = captures.get(1) {
                        if let Ok(size) = size_str.as_str().parse::<u32>() {
                            // HTX v1.1 spec: MAX_FRAME_SIZE = 2^24 - 1 = 16,777,215
                            if size != 16_777_215 {
                                issues.push(LintIssue::new(
                                    "FRAME001".to_string(),
                                    SeverityLevel::Error,
                                    format!("MAX_FRAME_SIZE {} incorrect - HTX v1.1 spec requires 16,777,215 (2^24-1)", size),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        // Check for MAX_STREAM_ID constant
        if context.content.contains("MAX_STREAM_ID") {
            let max_stream_regex = Regex::new(r"MAX_STREAM_ID\s*:\s*u32\s*=\s*(\d+)").unwrap();
            for (line_num, line) in context.content.lines().enumerate() {
                if let Some(captures) = max_stream_regex.captures(line) {
                    if let Some(id_str) = captures.get(1) {
                        if let Ok(id) = id_str.as_str().parse::<u32>() {
                            // HTX v1.1 spec: MAX_STREAM_ID = 2^28 - 1 = 268,435,455 (varint limit)
                            if id != 268_435_455 {
                                issues.push(LintIssue::new(
                                    "FRAME002".to_string(),
                                    SeverityLevel::Error,
                                    format!("MAX_STREAM_ID {} incorrect - HTX v1.1 spec requires 268,435,455 (2^28-1)", id),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        // Check frame encoding structure compliance
        if context.content.contains("encode") && context.content.contains("Frame") {
            let required_components = [
                ("uint24", "length field"),
                ("put_u8.*>>.*16", "uint24 big-endian encoding"),
                ("varint", "stream ID encoding"),
                ("frame_type as u8", "frame type field"),
                ("payload", "payload data"),
            ];

            for (pattern, description) in &required_components {
                if !context.content.contains(pattern) && !Regex::new(pattern).unwrap().is_match(&context.content) {
                    issues.push(LintIssue::new(
                        "FRAME003".to_string(),
                        SeverityLevel::Error,
                        format!("Frame encoding missing required component: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check frame parsing structure compliance
        if context.content.contains("parse_frame") || context.content.contains("decode") {
            let required_parsing = [
                ("data.len.*<.*4", "minimum header size check"),
                ("data\\[0\\].*<<.*16", "uint24 big-endian decoding"),
                ("decode_varint", "stream ID decoding"),
                ("FrameType::try_from", "frame type validation"),
            ];

            for (pattern, description) in &required_parsing {
                if !Regex::new(pattern).unwrap().is_match(&context.content) {
                    issues.push(LintIssue::new(
                        "FRAME004".to_string(),
                        SeverityLevel::Error,
                        format!("Frame parsing missing required check: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        Ok(issues)
    }
}

/// HTX frame type compliance
pub struct FrameTypeRule;

#[async_trait]
impl CheckRule for FrameTypeRule {
    fn name(&self) -> &str {
        "frame-type"
    }

    fn description(&self) -> &str {
        "Check HTX frame type definitions and validation"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check files with FrameType definitions
        if !context.content.contains("FrameType") {
            return Ok(issues);
        }

        // Check for required frame types per HTX v1.1 spec
        let required_frame_types = [
            ("Data", "0x00"),
            ("WindowUpdate", "0x01"),
            ("KeyUpdate", "0x02"),
            ("Ping", "0x03"),
            ("Priority", "0x04"),
            ("Padding", "0x05"),
            ("AccessTicket", "0x06"),
            ("Control", "0x07"),
        ];

        for (frame_type, expected_value) in &required_frame_types {
            if context.content.contains(&format!("enum FrameType")) {
                if !context.content.contains(frame_type) {
                    issues.push(LintIssue::new(
                        "FRAME005".to_string(),
                        SeverityLevel::Error,
                        format!("Missing required frame type: {}", frame_type),
                        self.name().to_string(),
                    ));
                } else {
                    // Check for correct value assignment
                    let type_regex = Regex::new(&format!(r"{}\s*=\s*(0x[0-9a-fA-F]+)", frame_type)).unwrap();
                    let mut found_correct_value = false;
                    for line in context.content.lines() {
                        if let Some(captures) = type_regex.captures(line) {
                            if let Some(value) = captures.get(1) {
                                if value.as_str() == *expected_value {
                                    found_correct_value = true;
                                } else {
                                    issues.push(LintIssue::new(
                                        "FRAME006".to_string(),
                                        SeverityLevel::Error,
                                        format!("Frame type {} has incorrect value {} - should be {}",
                                               frame_type, value.as_str(), expected_value),
                                        self.name().to_string(),
                                    ));
                                }
                                break;
                            }
                        }
                    }

                    if !found_correct_value && context.content.contains(&format!("{} =", frame_type)) {
                        issues.push(LintIssue::new(
                            "FRAME007".to_string(),
                            SeverityLevel::Warning,
                            format!("Could not verify {} frame type value - should be {}", frame_type, expected_value),
                            self.name().to_string(),
                        ));
                    }
                }
            }
        }

        // Check for proper TryFrom implementation
        if context.content.contains("TryFrom<u8>") && context.content.contains("FrameType") {
            // Verify all frame types are handled in try_from
            for (frame_type, expected_value) in &required_frame_types {
                let try_from_pattern = format!(r"{}\s*=>\s*Ok\(Self::{}\)", expected_value, frame_type);
                if !Regex::new(&try_from_pattern).unwrap().is_match(&context.content) {
                    issues.push(LintIssue::new(
                        "FRAME008".to_string(),
                        SeverityLevel::Error,
                        format!("TryFrom<u8> missing case for {} ({})", frame_type, expected_value),
                        self.name().to_string(),
                    ));
                }
            }

            // Check for proper error handling
            if !context.content.contains("InvalidFrameType") {
                issues.push(LintIssue::new(
                    "FRAME009".to_string(),
                    SeverityLevel::Error,
                    "TryFrom<u8> missing InvalidFrameType error handling".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// HTX varint encoding compliance
pub struct VarintEncodingRule;

#[async_trait]
impl CheckRule for VarintEncodingRule {
    fn name(&self) -> &str {
        "varint-encoding"
    }

    fn description(&self) -> &str {
        "Check varint (LEB128) encoding/decoding compliance"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check varint-related code
        if !context.content.contains("varint") &&
           !context.content.contains("LEB128") &&
           !context.content.contains("encode_varint") {
            return Ok(issues);
        }

        // Check varint encoding implementation
        if context.content.contains("encode_varint") {
            let encoding_checks = [
                ("value >= 0x80", "continuation bit check"),
                ("value & 0x7F", "7-bit data extraction"),
                ("0x80", "continuation bit setting"),
                ("value >>= 7", "7-bit shift"),
            ];

            for (pattern, description) in &encoding_checks {
                if !context.content.contains(pattern) {
                    issues.push(LintIssue::new(
                        "FRAME010".to_string(),
                        SeverityLevel::Error,
                        format!("Varint encoding missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check varint decoding implementation
        if context.content.contains("decode_varint") {
            let decoding_checks = [
                ("byte & 0x7F", "7-bit data extraction"),
                ("byte & 0x80", "continuation bit check"),
                ("shift += 7", "bit shift accumulation"),
                ("shift >= 28", "varint length limit"),
            ];

            for (pattern, description) in &decoding_checks {
                if !context.content.contains(pattern) {
                    issues.push(LintIssue::new(
                        "FRAME011".to_string(),
                        SeverityLevel::Error,
                        format!("Varint decoding missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check varint length calculation
        if context.content.contains("varint_length") {
            let length_boundaries = ["0x80", "0x4000", "0x200000"];
            for boundary in &length_boundaries {
                if !context.content.contains(boundary) {
                    issues.push(LintIssue::new(
                        "FRAME012".to_string(),
                        SeverityLevel::Warning,
                        format!("Varint length calculation missing boundary check: {}", boundary),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for proper error handling
        if context.content.contains("decode_varint") {
            if !context.content.contains("InvalidVarint") {
                issues.push(LintIssue::new(
                    "FRAME013".to_string(),
                    SeverityLevel::Error,
                    "Varint decoding missing InvalidVarint error handling".to_string(),
                    self.name().to_string(),
                ));
            }

            if !context.content.contains("StreamIdTooLarge") {
                issues.push(LintIssue::new(
                    "FRAME014".to_string(),
                    SeverityLevel::Error,
                    "Varint decoding missing StreamIdTooLarge validation".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// HTX frame buffer management compliance
pub struct FrameBufferRule;

#[async_trait]
impl CheckRule for FrameBufferRule {
    fn name(&self) -> &str {
        "frame-buffer"
    }

    fn description(&self) -> &str {
        "Check frame buffer management and zero-copy parsing"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check frame buffer related code
        if !context.content.contains("FrameBuffer") &&
           !context.content.contains("BytesMut") &&
           !context.content.contains("frame") {
            return Ok(issues);
        }

        // Check for proper buffer management
        if context.content.contains("FrameBuffer") {
            let buffer_features = [
                ("max_buffer_size", "buffer size limits"),
                ("append_data", "data accumulation"),
                ("parse_frames", "frame parsing"),
                ("advance", "consumed data removal"),
            ];

            for (feature, description) in &buffer_features {
                if !context.content.contains(feature) {
                    issues.push(LintIssue::new(
                        "FRAME015".to_string(),
                        SeverityLevel::Warning,
                        format!("FrameBuffer missing feature: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for zero-copy optimization
        if context.content.contains("parse") || context.content.contains("decode") {
            if context.content.contains("clone()") || context.content.contains("to_vec()") {
                // Check if it's in a context where copying might be unnecessary
                let copy_contexts = ["payload", "frame", "decode"];
                for context_word in &copy_contexts {
                    if context.content.contains(&format!("{}.clone()", context_word)) ||
                       context.content.contains(&format!("{}.to_vec()", context_word)) {
                        issues.push(LintIssue::new(
                            "FRAME016".to_string(),
                            SeverityLevel::Warning,
                            format!("Potential unnecessary copy in {} - consider zero-copy with Bytes", context_word),
                            self.name().to_string(),
                        ));
                    }
                }
            }

            // Check for proper use of BytesMut/Bytes
            if !context.content.contains("BytesMut") && !context.content.contains("Bytes") {
                issues.push(LintIssue::new(
                    "FRAME017".to_string(),
                    SeverityLevel::Warning,
                    "Consider using BytesMut/Bytes for zero-copy frame parsing".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check buffer overflow protection
        if context.content.contains("append_data") {
            if !context.content.contains("max_buffer_size") && !context.content.contains("buffer.len()") {
                issues.push(LintIssue::new(
                    "FRAME018".to_string(),
                    SeverityLevel::Error,
                    "Buffer append missing overflow protection".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for incomplete frame handling
        if context.content.contains("parse_frames") {
            if !context.content.contains("IncompleteFrame") {
                issues.push(LintIssue::new(
                    "FRAME019".to_string(),
                    SeverityLevel::Error,
                    "Frame parsing missing IncompleteFrame error handling".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// HTX frame validation compliance
pub struct FrameValidationRule;

#[async_trait]
impl CheckRule for FrameValidationRule {
    fn name(&self) -> &str {
        "frame-validation"
    }

    fn description(&self) -> &str {
        "Check frame validation and error handling compliance"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check frame validation code
        if !context.content.contains("Frame") &&
           !context.content.contains("validate") &&
           !context.content.contains("new") {
            return Ok(issues);
        }

        // Check DATA frame validation
        if context.content.contains("Frame::data") || context.content.contains("FrameType::Data") {
            if !context.content.contains("stream_id == 0") && !context.content.contains("stream_id > 0") {
                issues.push(LintIssue::new(
                    "FRAME020".to_string(),
                    SeverityLevel::Error,
                    "DATA frame missing stream_id > 0 validation".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check WINDOW_UPDATE frame validation
        if context.content.contains("window_update") || context.content.contains("WindowUpdate") {
            let window_checks = [
                ("window_delta == 0", "zero delta check"),
                ("0x7FFF_FFFF", "maximum delta check"),
            ];

            for (pattern, description) in &window_checks {
                if !context.content.contains(pattern) {
                    issues.push(LintIssue::new(
                        "FRAME021".to_string(),
                        SeverityLevel::Warning,
                        format!("WINDOW_UPDATE frame missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check PING frame validation
        if context.content.contains("ping") || context.content.contains("Ping") {
            if context.content.contains("ping_data") && !context.content.contains("data.len() > 8") {
                issues.push(LintIssue::new(
                    "FRAME022".to_string(),
                    SeverityLevel::Error,
                    "PING frame missing payload size validation (8 bytes max)".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check size validation
        if context.content.contains("Frame::new") {
            let size_checks = [
                ("MAX_FRAME_SIZE", "frame size limit"),
                ("MAX_STREAM_ID", "stream ID limit"),
                ("FrameTooLarge", "size error handling"),
                ("StreamIdTooLarge", "stream ID error handling"),
            ];

            for (pattern, description) in &size_checks {
                if !context.content.contains(pattern) {
                    issues.push(LintIssue::new(
                        "FRAME023".to_string(),
                        SeverityLevel::Warning,
                        format!("Frame creation missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check error enum completeness
        if context.content.contains("enum FrameError") {
            let required_errors = [
                "InvalidFrameType",
                "FrameTooLarge",
                "StreamIdTooLarge",
                "FrameTooShort",
                "InvalidVarint",
                "IncompleteFrame",
            ];

            for error_type in &required_errors {
                if !context.content.contains(error_type) {
                    issues.push(LintIssue::new(
                        "FRAME024".to_string(),
                        SeverityLevel::Warning,
                        format!("FrameError enum missing: {}", error_type),
                        self.name().to_string(),
                    ));
                }
            }
        }

        Ok(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_context(content: &str) -> CheckContext {
        CheckContext {
            file_path: PathBuf::from("test_frame.rs"),
            content: content.to_string(),
        }
    }

    #[tokio::test]
    async fn test_frame_structure_rule() {
        let rule = FrameStructureRule;

        // Test incorrect MAX_FRAME_SIZE
        let content = r#"
            pub const MAX_FRAME_SIZE: usize = 1048576; // Wrong value
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "FRAME001");

        // Test correct value
        let content = r#"
            pub const MAX_FRAME_SIZE: usize = 16777215; // Correct value
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 0);
    }

    #[tokio::test]
    async fn test_frame_type_rule() {
        let rule = FrameTypeRule;

        // Test missing frame type
        let content = r#"
            enum FrameType {
                Data = 0x00,
                // Missing WindowUpdate
                Ping = 0x03,
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|i| i.id == "FRAME005"));
    }

    #[tokio::test]
    async fn test_varint_encoding_rule() {
        let rule = VarintEncodingRule;

        // Test missing varint components
        let content = r#"
            fn encode_varint(value: u32) -> Vec<u8> {
                // Missing proper LEB128 encoding
                vec![value as u8]
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|i| i.id == "FRAME010"));
    }

    #[tokio::test]
    async fn test_frame_validation_rule() {
        let rule = FrameValidationRule;

        // Test missing DATA frame validation
        let content = r#"
            impl Frame {
                pub fn data(stream_id: u32, payload: Bytes) -> Result<Self, FrameError> {
                    // Missing stream_id > 0 check
                    Self::new(FrameType::Data, stream_id, payload)
                }
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "FRAME020");
    }

    #[tokio::test]
    async fn test_frame_buffer_rule() {
        let rule = FrameBufferRule;

        // Test missing buffer overflow protection
        let content = r#"
            impl FrameBuffer {
                pub fn append_data(&mut self, data: &[u8]) {
                    // Missing size check
                    self.buffer.extend_from_slice(data);
                }
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "FRAME018");
    }
}
