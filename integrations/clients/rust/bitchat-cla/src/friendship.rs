//! Friendship queues for Low Power Node (LPN) support in BitChat mesh
//!
//! Implements store-and-forward functionality for mobile devices in low power mode.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Maximum messages stored per friend
const MAX_MESSAGES_PER_FRIEND: usize = 50;

/// Maximum message age before cleanup
const MAX_MESSAGE_AGE: Duration = Duration::from_secs(3600); // 1 hour

/// Friendship configuration
#[derive(Debug, Clone)]
pub struct FriendshipConfig {
    /// Maximum number of friends supported
    pub max_friends: usize,
    /// Maximum messages per friend queue
    pub max_messages_per_friend: usize,
    /// Message retention duration
    pub message_retention: Duration,
    /// Friendship timeout
    pub friendship_timeout: Duration,
}

impl Default for FriendshipConfig {
    fn default() -> Self {
        Self {
            max_friends: 10,
            max_messages_per_friend: MAX_MESSAGES_PER_FRIEND,
            message_retention: MAX_MESSAGE_AGE,
            friendship_timeout: Duration::from_secs(86400), // 24 hours
        }
    }
}

/// Friend device information
#[derive(Debug, Clone)]
pub struct FriendDevice {
    pub device_id: String,
    pub last_seen: Instant,
    pub is_low_power: bool,
    pub poll_interval: Duration,
    pub message_queue: VecDeque<StoredMessage>,
    pub established_at: Instant,
}

/// Message stored for friend
#[derive(Debug, Clone)]
pub struct StoredMessage {
    pub message_id: String,
    pub payload: Vec<u8>,
    pub priority: MessagePriority,
    pub stored_at: Instant,
    pub sender: String,
    pub attempts: u32,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Friendship management engine
pub struct FriendshipEngine {
    config: FriendshipConfig,
    friends: HashMap<String, FriendDevice>,
    friendship_requests: HashMap<String, Instant>,
}

impl FriendshipEngine {
    pub fn new(config: FriendshipConfig) -> Self {
        Self {
            config,
            friends: HashMap::new(),
            friendship_requests: HashMap::new(),
        }
    }

    /// Request friendship with a device
    pub fn request_friendship(
        &mut self,
        device_id: String,
        is_low_power: bool,
        poll_interval: Duration,
    ) -> Result<(), FriendshipError> {
        if self.friends.len() >= self.config.max_friends {
            return Err(FriendshipError::MaxFriendsReached);
        }

        if self.friends.contains_key(&device_id) {
            return Err(FriendshipError::FriendshipExists);
        }

        // Record friendship request
        self.friendship_requests.insert(device_id.clone(), Instant::now());

        // For now, auto-accept friendship requests
        self.establish_friendship(device_id, is_low_power, poll_interval)?;

        Ok(())
    }

    /// Establish friendship with a device
    fn establish_friendship(
        &mut self,
        device_id: String,
        is_low_power: bool,
        poll_interval: Duration,
    ) -> Result<(), FriendshipError> {
        let friend = FriendDevice {
            device_id: device_id.clone(),
            last_seen: Instant::now(),
            is_low_power,
            poll_interval,
            message_queue: VecDeque::new(),
            established_at: Instant::now(),
        };

        self.friends.insert(device_id.clone(), friend);
        self.friendship_requests.remove(&device_id);

        Ok(())
    }

    /// Store message for a friend
    pub fn store_message_for_friend(
        &mut self,
        friend_id: &str,
        message: StoredMessage,
    ) -> Result<(), FriendshipError> {
        let friend = self.friends.get_mut(friend_id)
            .ok_or_else(|| FriendshipError::FriendNotFound(friend_id.to_string()))?;

        // Check queue capacity
        if friend.message_queue.len() >= self.config.max_messages_per_friend {
            // Remove oldest low-priority message to make space
            if let Some(pos) = friend.message_queue.iter().position(|msg| {
                msg.priority == MessagePriority::Low || msg.priority == MessagePriority::Normal
            }) {
                friend.message_queue.remove(pos);
            } else {
                return Err(FriendshipError::QueueFull);
            }
        }

        // Insert message maintaining priority order
        let insert_pos = friend.message_queue
            .iter()
            .position(|msg| msg.priority < message.priority)
            .unwrap_or(friend.message_queue.len());

        friend.message_queue.insert(insert_pos, message);

        Ok(())
    }

    /// Get pending messages for a friend (polling)
    pub fn get_pending_messages(&mut self, friend_id: &str) -> Result<Vec<StoredMessage>, FriendshipError> {
        let friend = self.friends.get_mut(friend_id)
            .ok_or_else(|| FriendshipError::FriendNotFound(friend_id.to_string()))?;

        // Update last seen time
        friend.last_seen = Instant::now();

        // Return all pending messages and clear queue
        let messages: Vec<StoredMessage> = friend.message_queue.drain(..).collect();

        Ok(messages)
    }

    /// Check if device is a friend
    pub fn is_friend(&self, device_id: &str) -> bool {
        self.friends.contains_key(device_id)
    }
}

/// Friendship errors
#[derive(Debug, thiserror::Error)]
pub enum FriendshipError {
    #[error("Maximum friends limit reached")]
    MaxFriendsReached,
    
    #[error("Friendship already exists")]
    FriendshipExists,
    
    #[error("Friend not found: {0}")]
    FriendNotFound(String),
    
    #[error("Message queue is full")]
    QueueFull,
    
    #[error("Invalid friendship request")]
    InvalidRequest,
}
