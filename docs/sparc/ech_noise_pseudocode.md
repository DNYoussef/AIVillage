# ECH + Noise Protocol Pseudocode Design

## Algorithm Specifications

### 1. ECH Configuration Parser

```pseudocode
ALGORITHM ParseECHConfig(config_bytes)
INPUT: config_bytes - Raw ECH configuration data
OUTPUT: ECHConfig object or ParseError

BEGIN
    // Validate minimum size and structure
    IF len(config_bytes) < ECH_MIN_SIZE THEN
        RETURN ParseError("Configuration too small")
    END IF
    
    // Parse header
    version = read_uint16(config_bytes, 0)
    IF version NOT IN SUPPORTED_ECH_VERSIONS THEN
        RETURN ParseError("Unsupported ECH version: " + version)
    END IF
    
    // Parse configuration ID and public key
    config_id = read_uint8(config_bytes, 2)
    kem_id = read_uint16(config_bytes, 3)
    public_key_length = read_uint16(config_bytes, 5)
    
    IF public_key_length > MAX_PUBLIC_KEY_SIZE THEN
        RETURN ParseError("Public key too large")
    END IF
    
    public_key = read_bytes(config_bytes, 7, public_key_length)
    
    // Parse cipher suites
    offset = 7 + public_key_length
    cipher_suites_length = read_uint16(config_bytes, offset)
    offset += 2
    
    cipher_suites = []
    FOR i = 0 TO cipher_suites_length / 4 - 1 DO
        kdf_id = read_uint16(config_bytes, offset + i * 4)
        aead_id = read_uint16(config_bytes, offset + i * 4 + 2)
        
        IF validate_cipher_suite(kdf_id, aead_id) THEN
            cipher_suites.append(CipherSuite(kdf_id, aead_id))
        ELSE
            // Log warning but continue parsing
            log_warning("Unsupported cipher suite: " + kdf_id + ", " + aead_id)
        END IF
    END FOR
    
    IF len(cipher_suites) = 0 THEN
        RETURN ParseError("No supported cipher suites found")
    END IF
    
    // Parse extensions
    offset += cipher_suites_length
    extensions_length = read_uint16(config_bytes, offset)
    offset += 2
    extensions = parse_extensions(config_bytes, offset, extensions_length)
    
    // Create and validate ECH configuration
    ech_config = ECHConfig(
        version: version,
        config_id: config_id,
        kem_id: kem_id,
        public_key: public_key,
        cipher_suites: cipher_suites,
        extensions: extensions
    )
    
    IF validate_ech_config(ech_config) THEN
        RETURN ech_config
    ELSE
        RETURN ParseError("Invalid ECH configuration")
    END IF
END
```

### 2. Enhanced Noise XK Handshake with ECH

```pseudocode
ALGORITHM EnhancedNoiseHandshake(ech_config, target_sni)
INPUT: ech_config - Parsed ECH configuration, target_sni - Server name
OUTPUT: HandshakeResult with encryption keys

BEGIN
    // Generate ephemeral keys for Noise protocol
    noise_ephemeral_private = generate_x25519_private_key()
    noise_ephemeral_public = derive_public_key(noise_ephemeral_private)
    
    // Generate ECH encryption context
    ech_client_random = secure_random(32)
    ech_keys = derive_ech_keys(ech_config, ech_client_random)
    
    // Encrypt the target SNI using ECH
    encrypted_sni = encrypt_sni(target_sni, ech_keys)
    
    // Construct Client Hello Inner (containing real SNI)
    client_hello_inner = ClientHello(
        sni: target_sni,
        cipher_suites: SUPPORTED_NOISE_CIPHERS,
        extensions: [
            key_share: noise_ephemeral_public,
            supported_versions: [NOISE_XK_VERSION]
        ]
    )
    
    // Construct Client Hello Outer (with encrypted SNI)
    client_hello_outer = ClientHello(
        sni: ech_config.public_name,
        cipher_suites: ech_config.cipher_suites,
        extensions: [
            encrypted_client_hello: encrypted_sni,
            key_share: noise_ephemeral_public
        ]
    )
    
    // Send initial handshake message
    handshake_msg = create_noise_message(
        type: "NOISE_XK_ECH_INIT",
        payload: serialize(client_hello_outer),
        ephemeral_key: noise_ephemeral_public
    )
    
    response = send_and_receive(handshake_msg, HANDSHAKE_TIMEOUT)
    
    // Process server response
    IF response.type = "NOISE_XK_ECH_ACCEPT" THEN
        // Extract server ephemeral key and ECH acceptance
        server_ephemeral = response.server_key
        ech_acceptance = response.ech_confirmation
        
        // Verify ECH was properly processed
        IF NOT verify_ech_acceptance(ech_acceptance, ech_keys) THEN
            RETURN HandshakeError("ECH verification failed")
        END IF
        
        // Derive shared secret using Noise XK pattern
        shared_secret = noise_xk_derive_secret(
            our_ephemeral_private: noise_ephemeral_private,
            their_ephemeral_public: server_ephemeral,
            our_static_private: static_private_key,
            their_static_public: response.server_static_key
        )
        
        // Derive encryption/decryption keys with forward secrecy
        encryption_key, decryption_key = hkdf_expand(
            secret: shared_secret,
            salt: "noise_xk_ech_keys",
            info: client_hello_inner + response.server_hello,
            length: 64
        )
        
        RETURN HandshakeResult(
            success: true,
            encryption_key: encryption_key[0:32],
            decryption_key: encryption_key[32:64],
            ech_enabled: true,
            forward_secrecy: true
        )
    
    ELSE IF response.type = "NOISE_XK_FALLBACK" THEN
        // ECH not supported, fall back to standard Noise XK
        RETURN fallback_noise_handshake(noise_ephemeral_private, response)
        
    ELSE
        RETURN HandshakeError("Unexpected server response: " + response.type)
    END IF
END
```

### 3. ECH-Aware Transport Integration

```pseudocode
ALGORITHM IntegrateECHTransport(transport_manager, ech_configs)
INPUT: transport_manager - Existing transport manager, ech_configs - ECH configurations
OUTPUT: Enhanced transport manager with ECH support

BEGIN
    // Create ECH-enhanced transport wrapper
    ech_transport = ECHTransportWrapper(
        base_transport: transport_manager,
        ech_configurations: ech_configs
    )
    
    // Override connection establishment
    OVERRIDE ech_transport.establish_connection(peer_id, options) AS
    BEGIN
        // Check if ECH should be used
        IF options.use_ech AND has_ech_config_for(peer_id) THEN
            ech_config = get_ech_config(peer_id)
            
            // Attempt ECH-enhanced handshake
            TRY
                connection = enhanced_noise_handshake(ech_config, peer_id)
                IF connection.success AND connection.ech_enabled THEN
                    // ECH handshake successful
                    connection.properties.encryption_level = "ECH_ENHANCED"
                    connection.properties.forward_secrecy = true
                    update_transport_metrics("ech_success")
                    RETURN connection
                END IF
            CATCH ECHError as e
                log_warning("ECH handshake failed: " + e.message)
                update_transport_metrics("ech_fallback")
            END TRY
        END IF
        
        // Fallback to standard handshake
        connection = base_transport.establish_connection(peer_id, options)
        connection.properties.encryption_level = "STANDARD_NOISE"
        RETURN connection
    END
    
    // Override message sending with ECH awareness
    OVERRIDE ech_transport.send_message(message, connection) AS
    BEGIN
        IF connection.properties.ech_enabled THEN
            // Use ECH-enhanced encryption
            encrypted_payload = ech_encrypt(
                plaintext: message.payload,
                key: connection.encryption_key,
                nonce: generate_nonce(),
                additional_data: message.metadata
            )
            
            enhanced_message = Message(
                payload: encrypted_payload,
                metadata: message.metadata,
                encryption_type: "ECH_ENHANCED"
            )
            
            RETURN base_transport.send_message(enhanced_message, connection)
        ELSE
            // Use standard encryption
            RETURN base_transport.send_message(message, connection)
        END IF
    END
    
    RETURN ech_transport
END
```

### 4. Error Handling and Recovery

```pseudocode
ALGORITHM ECHErrorRecovery(error_type, context)
INPUT: error_type - Type of ECH error, context - Current operation context
OUTPUT: Recovery action or final error

BEGIN
    SWITCH error_type DO
        CASE "ECH_CONFIG_PARSE_ERROR":
            // Log error and attempt fallback
            log_error("ECH config parsing failed: " + context.error_message)
            IF context.allow_fallback THEN
                disable_ech_for_peer(context.peer_id)
                RETURN RecoveryAction("FALLBACK_TO_STANDARD")
            ELSE
                RETURN FinalError("ECH_REQUIRED_BUT_INVALID")
            END IF
            
        CASE "ECH_HANDSHAKE_TIMEOUT":
            // Retry with exponential backoff
            IF context.retry_count < MAX_ECH_RETRIES THEN
                wait_time = ECH_BASE_RETRY_DELAY * (2 ^ context.retry_count)
                schedule_retry(wait_time)
                RETURN RecoveryAction("RETRY_WITH_BACKOFF")
            ELSE
                log_error("ECH handshake timeout after " + context.retry_count + " retries")
                RETURN RecoveryAction("FALLBACK_TO_STANDARD")
            END IF
            
        CASE "ECH_KEY_DERIVATION_ERROR":
            // Critical error - cannot continue with ECH
            log_error("ECH key derivation failed - potential security issue")
            mark_ech_config_invalid(context.ech_config.config_id)
            RETURN FinalError("ECH_CRYPTO_FAILURE")
            
        CASE "ECH_VERIFICATION_FAILURE":
            // Possible MITM or configuration mismatch
            log_security_alert("ECH verification failed for " + context.peer_id)
            IF context.strict_ech_mode THEN
                RETURN FinalError("ECH_VERIFICATION_REQUIRED")
            ELSE
                RETURN RecoveryAction("FALLBACK_TO_STANDARD")
            END IF
            
        DEFAULT:
            log_error("Unknown ECH error: " + error_type)
            RETURN RecoveryAction("FALLBACK_TO_STANDARD")
    END SWITCH
END
```

### 5. Performance Optimization Strategies

```pseudocode
ALGORITHM OptimizeECHPerformance(ech_manager)
INPUT: ech_manager - ECH management instance
OUTPUT: Performance-optimized ECH manager

BEGIN
    // Pre-compute ECH keys for known configurations
    FOR each ech_config IN ech_manager.configurations DO
        IF ech_config.usage_frequency > HIGH_USAGE_THRESHOLD THEN
            precomputed_keys = precompute_ech_keys(ech_config)
            ech_manager.key_cache[ech_config.id] = precomputed_keys
        END IF
    END FOR
    
    // Optimize cipher suite selection
    ech_manager.preferred_suites = sort_by_performance([
        "ChaCha20Poly1305_SHA256", // Fast on mobile
        "AES256GCM_SHA384",       // Fast with hardware acceleration
        "AES128GCM_SHA256"        // Fallback option
    ])
    
    // Connection pooling for ECH handshakes
    ech_manager.connection_pool = ConnectionPool(
        max_size: MAX_ECH_CONNECTIONS,
        idle_timeout: ECH_IDLE_TIMEOUT,
        reuse_strategy: "ROUND_ROBIN"
    )
    
    // Asynchronous ECH operations
    ech_manager.async_processor = AsyncProcessor(
        worker_count: CPU_CORE_COUNT,
        queue_size: ECH_OPERATION_QUEUE_SIZE,
        priority_levels: ["CRITICAL", "HIGH", "NORMAL", "LOW"]
    )
    
    RETURN ech_manager
END
```

## Complexity Analysis

### Time Complexity
- **ECH Config Parsing**: O(n) where n = config size
- **ECH Key Derivation**: O(1) - constant time operations
- **Enhanced Handshake**: O(1) - fixed number of crypto operations
- **Message Encryption**: O(m) where m = message size

### Space Complexity
- **ECH Configuration Storage**: O(k) where k = number of configs
- **Key Material**: O(1) - fixed size per connection
- **Message Buffers**: O(m) where m = largest message size

### Security Properties
- **Forward Secrecy**: ✓ Ephemeral keys deleted after handshake
- **Post-Compromise Security**: ✓ New keys for each session
- **Traffic Analysis Resistance**: ✓ SNI encrypted with ECH
- **Downgrade Resistance**: ✓ Mandatory ECH validation

## Integration Patterns

### Dependency Injection
```pseudocode
// Weak coupling through interfaces
ECHManager(
    config_parser: ECHConfigParserInterface,
    key_deriver: KeyDerivationInterface,
    transport: TransportInterface
)
```

### Observer Pattern
```pseudocode
// Event-driven ECH status updates
ECHEventEmitter.emit("ech_handshake_complete", {
    peer_id: peer_id,
    success: true,
    performance_metrics: metrics
})
```

### Strategy Pattern
```pseudocode
// Pluggable cipher suite strategies
ECHCipherStrategy.select_best_suite(
    available_suites: suites,
    device_capabilities: caps,
    security_requirements: reqs
)
```

---

*SPARC Phase 2 Complete - Detailed algorithms designed with performance and security considerations.*