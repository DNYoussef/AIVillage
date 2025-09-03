use betanet_dtn::{BundleStore, Bundle, EndpointId};
use bytes::Bytes;
use tempfile::tempdir;

#[tokio::test]
async fn durability_persists_across_restart() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("db");
    let src = EndpointId::node("node1");
    let dst = EndpointId::node("node2");
    let bundle = Bundle::new(dst, src, Bytes::from("persist"), 60000);
    let id = bundle.id();

    {
        let store = BundleStore::open(&path).await.unwrap();
        store.store(bundle).await.unwrap();
        store.flush().await.unwrap();
    }

    let reopened = BundleStore::open(&path).await.unwrap();
    assert!(reopened.get(&id).await.unwrap().is_some());
}

#[tokio::test]
async fn cleanup_removes_expired_and_orphaned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("db");
    let store = BundleStore::open(&path).await.unwrap();

    let src = EndpointId::node("node1");
    let dst = EndpointId::node("node2");
    let bundle = Bundle::new(dst.clone(), src.clone(), Bytes::from("test"), 1);
    let id = bundle.id();
    store.store(bundle).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(store.cleanup_expired().await.unwrap(), 1);
    assert!(store.get(&id).await.unwrap().is_none());

    // Insert orphan index directly
    let db = sled::open(&path).unwrap();
    let dest_tree = db.open_tree("by_destination").unwrap();
    dest_tree.insert(b"orphan#1", b"missing").unwrap();
    db.flush().unwrap();

    assert_eq!(store.cleanup_orphans().await.unwrap(), 1);
    assert!(dest_tree.get(b"orphan#1").unwrap().is_none());
}
