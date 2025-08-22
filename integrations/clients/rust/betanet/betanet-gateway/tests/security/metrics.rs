use std::fs;
use std::path::Path;
use std::sync::Arc;

use tokio::runtime::Runtime;

use betanet_gateway::aead::{AeadManager, FrameType};
use betanet_gateway::config::GatewayConfig;
use betanet_gateway::metrics::MetricsCollector;

#[test]
fn export_metrics_snapshot() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config.clone()).unwrap());
        let manager = AeadManager::new(gateway_config.aead.clone(), metrics.clone(), [0u8; 32]);

        let peer = "metrics_peer";
        let payload = b"metrics";
        let aad = b"";

        let frame = manager
            .encrypt_frame(peer, FrameType::ScionData, payload, aad)
            .await
            .unwrap();
        manager.decrypt_frame(peer, &frame).await.unwrap();

        let enc = metrics
            .aead_encryptions
            .with_label_values(&["scion_data", "success"])
            .get();
        let dec = metrics
            .aead_decryptions
            .with_label_values(&["scion_data", "success"])
            .get();
        let auth_fail = metrics.aead_auth_failures.get();

        assert!(enc > 0);
        assert!(dec > 0);
        assert_eq!(auth_fail, 0);

        let out_dir = Path::new("../tmp_scion_perf");
        fs::create_dir_all(out_dir).unwrap();
        metrics
            .export_to_file(out_dir.join("metrics_snapshot.prom"))
            .unwrap();
    });
}
