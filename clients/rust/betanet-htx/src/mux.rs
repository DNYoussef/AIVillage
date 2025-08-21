//! Stream multiplexer with weighted round-robin scheduling
//!
//! Implements multiplexed streams over a single transport connection with:
//! - Weighted round-robin scheduler for fair bandwidth allocation
//! - WINDOW_UPDATE flow control and backpressure management
//! - Stream ID management (odd for client, even for server)

use crate::{Frame, FrameType, HtxError, Result};
use bytes::{Bytes, BytesMut};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::debug;

/// Default flow control window size
const DEFAULT_WINDOW_SIZE: u32 = 65536;
/// Minimum window size before triggering WINDOW_UPDATE
const MIN_WINDOW_SIZE: u32 = 8192;
/// Maximum number of concurrent streams
const MAX_CONCURRENT_STREAMS: usize = 1000;

/// Stream multiplexer with weighted round-robin scheduling
pub struct StreamMux {
    /// Stream state tracking
    streams: Arc<DashMap<u32, Arc<Mutex<MuxStream>>>>,
    /// Scheduler state
    scheduler: Arc<Mutex<WeightedRRScheduler>>,
    /// Flow control state
    flow_control: Arc<Mutex<GlobalFlowControl>>,
    /// Connection-level window
    connection_window: Arc<AtomicU32>,
    /// Next stream ID generator (odd for client, even for server)
    next_stream_id: Arc<AtomicU32>,
    /// Stream creation configuration
    is_client: bool,
    /// Frame output channel
    frame_sender: mpsc::UnboundedSender<Frame>,
    /// Statistics
    stats: Arc<Mutex<MuxStats>>,
}

/// Individual multiplexed stream
#[derive(Debug)]
pub struct MuxStream {
    /// Stream identifier
    pub stream_id: u32,
    /// Stream state
    pub state: StreamState,
    /// Send window size (flow control)
    pub send_window: u32,
    /// Receive window size
    pub receive_window: u32,
    /// Buffered outbound data
    pub send_buffer: VecDeque<Bytes>,
    /// Buffered inbound data
    pub receive_buffer: BytesMut,
    /// Stream priority/weight for scheduling
    pub weight: u32,
    /// Bytes sent on this stream
    pub bytes_sent: u64,
    /// Bytes received on this stream
    pub bytes_received: u64,
    /// Stream creation timestamp
    pub created_at: std::time::Instant,
}

/// Stream state enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is open and ready for data
    Open,
    /// Stream has been closed by local peer
    LocalClosed,
    /// Stream has been closed by remote peer
    RemoteClosed,
    /// Stream is fully closed
    Closed,
    /// Stream has encountered an error
    Error(String),
}

/// Weighted round-robin scheduler
#[derive(Debug)]
struct WeightedRRScheduler {
    /// Stream scheduling queue with weights
    stream_queue: VecDeque<ScheduledStream>,
    /// Current scheduling round weights
    current_weights: HashMap<u32, u32>,
    /// Total weight in current round
    total_weight: u32,
}

/// Scheduled stream entry
#[derive(Debug, Clone)]
struct ScheduledStream {
    stream_id: u32,
    weight: u32,
    remaining_weight: u32,
}

/// Global flow control state
#[derive(Debug)]
struct GlobalFlowControl {
    /// Total connection-level send window
    connection_send_window: u32,
    /// Connection-level receive window
    connection_receive_window: u32,
    /// Pending WINDOW_UPDATE frames
    pending_updates: HashMap<u32, u32>,
}

/// Multiplexer statistics
#[derive(Debug, Clone, Default)]
pub struct MuxStats {
    /// Total streams created
    pub streams_created: u64,
    /// Total streams closed
    pub streams_closed: u64,
    /// Active streams count
    pub active_streams: u32,
    /// Total bytes scheduled
    pub bytes_scheduled: u64,
    /// Total frames sent
    pub frames_sent: u64,
    /// Window updates sent
    pub window_updates_sent: u64,
    /// Backpressure events
    pub backpressure_events: u64,
}

impl StreamMux {
    /// Create new stream multiplexer
    pub fn new(is_client: bool, frame_sender: mpsc::UnboundedSender<Frame>) -> Self {
        let next_stream_id = if is_client { 1 } else { 2 };

        Self {
            streams: Arc::new(DashMap::new()),
            scheduler: Arc::new(Mutex::new(WeightedRRScheduler::new())),
            flow_control: Arc::new(Mutex::new(GlobalFlowControl::new())),
            connection_window: Arc::new(AtomicU32::new(DEFAULT_WINDOW_SIZE)),
            next_stream_id: Arc::new(AtomicU32::new(next_stream_id)),
            is_client,
            frame_sender,
            stats: Arc::new(Mutex::new(MuxStats::default())),
        }
    }

    /// Create new outbound stream
    pub async fn create_stream(&self, weight: Option<u32>) -> Result<u32> {
        // Check stream limit
        if self.streams.len() >= MAX_CONCURRENT_STREAMS {
            return Err(HtxError::Protocol(format!(
                "Maximum concurrent streams exceeded: {}",
                MAX_CONCURRENT_STREAMS
            )));
        }

        // Generate stream ID
        let stream_id = self.next_stream_id.fetch_add(2, Ordering::SeqCst);

        // Create stream
        let stream = MuxStream {
            stream_id,
            state: StreamState::Open,
            send_window: DEFAULT_WINDOW_SIZE,
            receive_window: DEFAULT_WINDOW_SIZE,
            send_buffer: VecDeque::new(),
            receive_buffer: BytesMut::new(),
            weight: weight.unwrap_or(100),
            bytes_sent: 0,
            bytes_received: 0,
            created_at: std::time::Instant::now(),
        };

        // Register with scheduler
        {
            let mut scheduler = self.scheduler.lock().await;
            scheduler.add_stream(stream_id, stream.weight);
        }

        // Insert stream
        self.streams
            .insert(stream_id, Arc::new(Mutex::new(stream)));

        // Update statistics
        {
            let mut stats = self.stats.lock().await;
            stats.streams_created += 1;
            stats.active_streams = self.streams.len() as u32;
        }

        debug!("Created stream {} with weight {}", stream_id, weight.unwrap_or(100));
        Ok(stream_id)
    }

    /// Send data on a stream
    pub async fn send_data(&self, stream_id: u32, data: Bytes) -> Result<()> {
        let stream_ref = self.streams.get(&stream_id).ok_or_else(|| {
            HtxError::Stream {
                stream_id,
                reason: "Stream not found".to_string(),
            }
        })?;

        let mut stream = stream_ref.lock().await;

        // Check stream state
        if !matches!(stream.state, StreamState::Open) {
            return Err(HtxError::Stream {
                stream_id,
                reason: format!("Stream in invalid state: {:?}", stream.state),
            });
        }

        // Check flow control
        let data_len = data.len();
        if stream.send_window < data_len as u32 {
            // Buffer data if window is insufficient
            stream.send_buffer.push_back(data);

            // Update backpressure stats
            let mut stats = self.stats.lock().await;
            stats.backpressure_events += 1;

            debug!("Buffering {} bytes for stream {} (window: {})",
                   data_len, stream_id, stream.send_window);
            return Ok(());
        }

        // Consume send window
        stream.send_window = stream.send_window.saturating_sub(data_len as u32);
        stream.bytes_sent += data_len as u64;

        // Create DATA frame
        let frame = Frame::data(stream_id, data)?;

        // Send frame
        self.frame_sender
            .send(frame)
            .map_err(|_| HtxError::Protocol("Frame channel closed".to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.lock().await;
            stats.frames_sent += 1;
            stats.bytes_scheduled += data_len as u64;
        }

        Ok(())
    }

    /// Process incoming frame
    pub async fn process_frame(&self, frame: Frame) -> Result<()> {
        match frame.frame_type {
            FrameType::Data => self.handle_data_frame(frame).await,
            FrameType::WindowUpdate => self.handle_window_update(frame).await,
            _ => {
                // Forward other frame types to application
                debug!("Forwarding frame type {:?} to application", frame.frame_type);
                Ok(())
            }
        }
    }

    /// Handle incoming DATA frame
    async fn handle_data_frame(&self, frame: Frame) -> Result<()> {
        let stream_id = frame.stream_id;

        // Get or create stream for incoming data
        let stream_ref = if let Some(stream_ref) = self.streams.get(&stream_id) {
            stream_ref.clone()
        } else {
            // Create new incoming stream if we're the server
            if self.is_client && stream_id % 2 == 0 {
                return Err(HtxError::Protocol(
                    "Client received server-initiated stream".to_string(),
                ));
            }
            if !self.is_client && stream_id % 2 == 1 {
                return Err(HtxError::Protocol(
                    "Server received client-initiated stream".to_string(),
                ));
            }

            let new_stream = MuxStream {
                stream_id,
                state: StreamState::Open,
                send_window: DEFAULT_WINDOW_SIZE,
                receive_window: DEFAULT_WINDOW_SIZE,
                send_buffer: VecDeque::new(),
                receive_buffer: BytesMut::new(),
                weight: 100,
                bytes_sent: 0,
                bytes_received: 0,
                created_at: std::time::Instant::now(),
            };

            let stream_ref = Arc::new(Mutex::new(new_stream));
            self.streams.insert(stream_id, stream_ref.clone());
            stream_ref
        };

        let mut stream = stream_ref.lock().await;

        // Update receive window
        let data_len = frame.payload.len() as u32;
        stream.receive_window = stream.receive_window.saturating_sub(data_len);
        stream.bytes_received += data_len as u64;

        // Buffer received data
        stream.receive_buffer.extend_from_slice(&frame.payload);

        // Send WINDOW_UPDATE if window is getting low
        if stream.receive_window < MIN_WINDOW_SIZE {
            let window_increment = DEFAULT_WINDOW_SIZE - stream.receive_window;
            stream.receive_window += window_increment;

            let window_update = Frame::window_update(stream_id, window_increment)?;
            self.frame_sender
                .send(window_update)
                .map_err(|_| HtxError::Protocol("Frame channel closed".to_string()))?;

            // Update statistics
            let mut stats = self.stats.lock().await;
            stats.window_updates_sent += 1;

            debug!("Sent WINDOW_UPDATE for stream {} (increment: {})",
                   stream_id, window_increment);
        }

        debug!("Received {} bytes on stream {} (window: {})",
               data_len, stream_id, stream.receive_window);
        Ok(())
    }

    /// Handle WINDOW_UPDATE frame
    async fn handle_window_update(&self, frame: Frame) -> Result<()> {
        if frame.payload.len() < 4 {
            return Err(HtxError::Protocol(
                "WINDOW_UPDATE payload too short".to_string(),
            ));
        }

        let window_increment = u32::from_be_bytes([
            frame.payload[0],
            frame.payload[1],
            frame.payload[2],
            frame.payload[3],
        ]);

        let stream_id = frame.stream_id;

        if stream_id == 0 {
            // Connection-level window update
            let old_window = self.connection_window.fetch_add(window_increment, Ordering::SeqCst);
            debug!("Connection window updated: {} -> {}", old_window, old_window + window_increment);
        } else {
            // Stream-level window update
            if let Some(stream_ref) = self.streams.get(&stream_id) {
                let mut stream = stream_ref.lock().await;
                stream.send_window = stream.send_window.saturating_add(window_increment);

                // Try to send buffered data
                while let Some(data) = stream.send_buffer.front().cloned() {
                    if stream.send_window >= data.len() as u32 {
                        stream.send_buffer.pop_front();
                        let data_len = data.len() as u32;
                        stream.send_window = stream.send_window.saturating_sub(data_len);
                        stream.bytes_sent += data_len as u64;

                        let frame = Frame::data(stream_id, data)?;
                        self.frame_sender
                            .send(frame)
                            .map_err(|_| HtxError::Protocol("Frame channel closed".to_string()))?;

                        // Update statistics
                        let mut stats = self.stats.lock().await;
                        stats.frames_sent += 1;
                        stats.bytes_scheduled += data_len as u64;
                    } else {
                        break;
                    }
                }

                debug!("Stream {} window updated: increment {}, new window: {}",
                       stream_id, window_increment, stream.send_window);
            }
        }

        Ok(())
    }

    /// Close a stream
    pub async fn close_stream(&self, stream_id: u32, local_close: bool) -> Result<()> {
        if let Some(stream_ref) = self.streams.get(&stream_id) {
            let mut stream = stream_ref.lock().await;

            stream.state = match (&stream.state, local_close) {
                (StreamState::Open, true) => StreamState::LocalClosed,
                (StreamState::Open, false) => StreamState::RemoteClosed,
                (StreamState::LocalClosed, false) => StreamState::Closed,
                (StreamState::RemoteClosed, true) => StreamState::Closed,
                _ => stream.state.clone(),
            };

            if matches!(stream.state, StreamState::Closed) {
                // Remove from scheduler
                let mut scheduler = self.scheduler.lock().await;
                scheduler.remove_stream(stream_id);

                // Update statistics
                let mut stats = self.stats.lock().await;
                stats.streams_closed += 1;
                stats.active_streams = stats.active_streams.saturating_sub(1);

                debug!("Stream {} fully closed", stream_id);
            }
        }

        Ok(())
    }

    /// Get multiplexer statistics
    pub async fn stats(&self) -> MuxStats {
        self.stats.lock().await.clone()
    }

    /// Get stream data if available
    pub async fn get_stream_data(&self, stream_id: u32) -> Option<Bytes> {
        if let Some(stream_ref) = self.streams.get(&stream_id) {
            let mut stream = stream_ref.lock().await;
            if !stream.receive_buffer.is_empty() {
                let data = stream.receive_buffer.split().freeze();
                return Some(data);
            }
        }
        None
    }

    /// List active streams
    pub fn active_stream_ids(&self) -> Vec<u32> {
        self.streams.iter().map(|entry| *entry.key()).collect()
    }
}

impl WeightedRRScheduler {
    /// Create new weighted round-robin scheduler
    fn new() -> Self {
        Self {
            stream_queue: VecDeque::new(),
            current_weights: HashMap::new(),
            total_weight: 0,
        }
    }

    /// Add stream to scheduler
    fn add_stream(&mut self, stream_id: u32, weight: u32) {
        self.stream_queue.push_back(ScheduledStream {
            stream_id,
            weight,
            remaining_weight: weight,
        });
        self.current_weights.insert(stream_id, weight);
        self.total_weight += weight;
    }

    /// Remove stream from scheduler
    fn remove_stream(&mut self, stream_id: u32) {
        self.stream_queue.retain(|s| s.stream_id != stream_id);
        if let Some(weight) = self.current_weights.remove(&stream_id) {
            self.total_weight = self.total_weight.saturating_sub(weight);
        }
    }

    /// Get next stream to schedule
    fn next_stream(&mut self) -> Option<u32> {
        if self.stream_queue.is_empty() {
            return None;
        }

        // Find stream with highest remaining weight
        let mut max_weight = 0;
        let mut selected_idx = None;

        for (idx, stream) in self.stream_queue.iter().enumerate() {
            if stream.remaining_weight > max_weight {
                max_weight = stream.remaining_weight;
                selected_idx = Some(idx);
            }
        }

        if let Some(idx) = selected_idx {
            let mut selected = self.stream_queue.remove(idx).unwrap();
            let stream_id = selected.stream_id;

            // Decrement remaining weight
            selected.remaining_weight = selected.remaining_weight.saturating_sub(1);

            // If weight remaining, add back to queue
            if selected.remaining_weight > 0 {
                self.stream_queue.push_back(selected);
            } else {
                // Reset weight for next round
                selected.remaining_weight = selected.weight;
                self.stream_queue.push_back(selected);
            }

            Some(stream_id)
        } else {
            None
        }
    }
}

impl GlobalFlowControl {
    /// Create new global flow control state
    fn new() -> Self {
        Self {
            connection_send_window: DEFAULT_WINDOW_SIZE,
            connection_receive_window: DEFAULT_WINDOW_SIZE,
            pending_updates: HashMap::new(),
        }
    }

    /// Update connection-level send window
    fn update_send_window(&mut self, increment: u32) {
        self.connection_send_window = self.connection_send_window.saturating_add(increment);
    }

    /// Consume connection-level send window
    fn consume_send_window(&mut self, amount: u32) -> bool {
        if self.connection_send_window >= amount {
            self.connection_send_window -= amount;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_stream_creation() {
        let (frame_sender, _frame_receiver) = mpsc::unbounded_channel();
        let mux = StreamMux::new(true, frame_sender);

        let stream_id = mux.create_stream(Some(200)).await.unwrap();
        assert_eq!(stream_id, 1); // First client stream

        let stream_id2 = mux.create_stream(None).await.unwrap();
        assert_eq!(stream_id2, 3); // Second client stream

        assert_eq!(mux.streams.len(), 2);
    }

    #[tokio::test]
    async fn test_server_stream_creation() {
        let (frame_sender, _frame_receiver) = mpsc::unbounded_channel();
        let mux = StreamMux::new(false, frame_sender);

        let stream_id = mux.create_stream(Some(150)).await.unwrap();
        assert_eq!(stream_id, 2); // First server stream

        let stream_id2 = mux.create_stream(None).await.unwrap();
        assert_eq!(stream_id2, 4); // Second server stream
    }

    #[tokio::test]
    async fn test_weighted_rr_scheduler() {
        let mut scheduler = WeightedRRScheduler::new();

        scheduler.add_stream(1, 100);
        scheduler.add_stream(2, 200);
        scheduler.add_stream(3, 50);

        // Stream 2 should be scheduled more frequently due to higher weight
        let mut stream_counts = HashMap::new();
        for _ in 0..350 {
            if let Some(stream_id) = scheduler.next_stream() {
                *stream_counts.entry(stream_id).or_insert(0) += 1;
            }
        }

        // Verify proportional scheduling
        assert!(stream_counts[&2] > stream_counts[&1]);
        assert!(stream_counts[&1] > stream_counts[&3]);
    }

    #[tokio::test]
    async fn test_flow_control() {
        let (frame_sender, mut frame_receiver) = mpsc::unbounded_channel();
        let mux = StreamMux::new(true, frame_sender);

        let stream_id = mux.create_stream(None).await.unwrap();
        let data = Bytes::from(vec![42u8; 1000]);

        // Send data
        mux.send_data(stream_id, data).await.unwrap();

        // Should receive a DATA frame
        let frame = frame_receiver.recv().await.unwrap();
        assert_eq!(frame.frame_type, FrameType::Data);
        assert_eq!(frame.stream_id, stream_id);
    }

    #[tokio::test]
    async fn test_window_update_handling() {
        let (frame_sender, _frame_receiver) = mpsc::unbounded_channel();
        let mux = StreamMux::new(true, frame_sender);

        let stream_id = mux.create_stream(None).await.unwrap();

        // Create WINDOW_UPDATE frame
        let window_increment = 32768u32;
        let window_update = Frame::window_update(stream_id, window_increment).unwrap();

        // Process the frame
        mux.process_frame(window_update).await.unwrap();

        // Verify window was updated
        let stream_ref = mux.streams.get(&stream_id).unwrap();
        let stream = stream_ref.lock().await;
        assert_eq!(stream.send_window, DEFAULT_WINDOW_SIZE + window_increment);
    }
}
