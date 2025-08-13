use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;
use tokio::runtime::Runtime;

use betanet_gateway::anti_replay::AntiReplayManager;
use betanet_gateway::config::{AntiReplayConfig, GatewayConfig};
use betanet_gateway::metrics::MetricsCollector;

#[test]
fn replay_persistence() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("replay.db");
        let config = AntiReplayConfig {
            db_path: db_path.clone(),
            window_size: 1024,
            cleanup_ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(300),
            sync_interval: Duration::from_secs(60),
            max_sequence_age: Duration::from_secs(300),
        };
        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config.clone()).unwrap());
        let manager = AntiReplayManager::new(config.clone(), metrics).await.unwrap();

        let peer = "peer1";
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // validate unique sequences
        for seq in 1..=100u64 {
            let res = manager.validate_sequence(peer, seq, timestamp, true).await;
            assert!(res.valid);
        }

        // inject duplicates
        let injected = vec![10u64, 20, 30];
        let mut drops = 0usize;
        for seq in &injected {
            let res = manager.validate_sequence(peer, *seq, timestamp, true).await;
            if !res.valid && res.rejection_reason == "replay" {
                drops += 1;
            }
        }

        // check for false reject
        let res = manager.validate_sequence(peer, 101, timestamp, true).await;
        let false_rejects = if res.valid { 0usize } else { 1usize };

        // drop manager and reopen to test persistence
        drop(manager);
        let metrics = Arc::new(MetricsCollector::new(gateway_config.clone()).unwrap());
        let manager = AntiReplayManager::new(config, metrics).await.unwrap();
        let res_persist = manager.validate_sequence(peer, injected[0], timestamp, true).await;
        assert!(!res_persist.valid);

        // log results
        let verify_dir = Path::new("../tmp_scion_verify");
        fs::create_dir_all(verify_dir).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(verify_dir.join("replay_persistence.log"))
            .unwrap();
        writeln!(file, "drops: {}, false_rejects: {}", drops, false_rejects).unwrap();

        assert_eq!(drops, injected.len());
        assert_eq!(false_rejects, 0);
    });
}
