use betanet_mixnode::{sphinx::serialize, MixNode};
use std::net::SocketAddr;
use tokio::net::UdpSocket;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn interop_route() {
    let recv_addr: SocketAddr = "127.0.0.1:50002".parse().unwrap();
    let mix_addr: SocketAddr = "127.0.0.1:50001".parse().unwrap();

    let receiver = UdpSocket::bind(recv_addr).await.unwrap();
    let mix = MixNode { bind: mix_addr, peers: vec![recv_addr] };
    tokio::spawn(async move {
        let _ = mix.run().await;
    });
    sleep(Duration::from_millis(100)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let packet = serialize(recv_addr, b"hello");
    client.send_to(&packet, mix_addr).await.unwrap();

    let mut buf = [0u8; 1500];
    let (n, _) = receiver.recv_from(&mut buf).await.unwrap();
    assert_eq!(&buf[..n], b"hello");
}
