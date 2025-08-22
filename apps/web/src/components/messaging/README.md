# BitChat P2P Messaging System

## Overview

BitChat is a decentralized peer-to-peer messaging system built into AIVillage that provides secure, encrypted communication without relying on central servers.

## Features

### 🔒 Security
- **End-to-End Encryption**: All messages are encrypted using AES-GCM
- **Secure Key Exchange**: Automatic key generation and exchange for each peer
- **No Central Servers**: Direct P2P connections ensure privacy

### 🌐 P2P Networking
- **WebRTC Data Channels**: Direct browser-to-browser communication
- **NAT Traversal**: STUN servers for connectivity across firewalls
- **Connection Management**: Automatic reconnection and status monitoring

### 💬 Messaging
- **Real-time Chat**: Instant message delivery over P2P connections
- **File Sharing**: Send files of any type directly to peers
- **Typing Indicators**: See when contacts are typing
- **Message Persistence**: Messages stored locally in browser

### 👥 Contact Management
- **Contact Discovery**: Find peers through signaling server
- **Peer Status**: Online/offline/connecting status indicators
- **Contact Profiles**: Name, avatar, and public key management

## Components

### 1. BitChatService (`bitchatService.ts`)
Core service managing P2P connections and messaging functionality.

**Key Methods:**
- `connectToPeer(peerId)` - Establish WebRTC connection
- `sendMessage(to, content)` - Send encrypted message
- `sendFile(to, file)` - Send file over P2P
- `addContact(contact)` - Add new contact
- `getMessages(peerId)` - Retrieve conversation history

**Events:**
- `message-received` - New message arrived
- `peer-connected` - Peer connection established
- `contact-updated` - Contact status changed
- `typing` - Typing indicator from peer

### 2. ContactsList Component
Manages contact list with discovery and connection features.

**Features:**
- Search and filter contacts
- Add contacts by Peer ID
- Connection status indicators
- Discovered peer suggestions
- Contact management (edit/remove)

### 3. ChatWindow Component
Main chat interface for conversations.

**Features:**
- Message bubbles with timestamps
- File upload and download
- Typing indicators
- Connection status display
- Encrypted message indicators

## Usage

### Basic Integration

```tsx
import ContactsList from './components/messaging/ContactsList';
import ChatWindow from './components/messaging/ChatWindow';
import { bitChatService } from './services/bitchatService';

function MessagingApp() {
  const [selectedContact, setSelectedContact] = useState(null);
  const [currentContact, setCurrentContact] = useState(null);

  const handleContactSelect = (contactId) => {
    setSelectedContact(contactId);
    const contact = bitChatService.getContacts().find(c => c.id === contactId);
    setCurrentContact(contact);
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ width: '300px' }}>
        <ContactsList
          selectedContact={selectedContact}
          onContactSelect={handleContactSelect}
          onNewMessage={(contact) => console.log('New message from:', contact.name)}
        />
      </div>
      <div style={{ flex: 1 }}>
        <ChatWindow
          contact={currentContact}
          onBack={() => setSelectedContact(null)}
        />
      </div>
    </div>
  );
}
```

### Adding a Contact

```tsx
// Add contact programmatically
bitChatService.addContact({
  id: 'peer_12345',
  name: 'Alice Johnson',
  publicKey: 'alice_public_key_here'
});
```

### Sending Messages

```tsx
// Send text message
await bitChatService.sendMessage('peer_12345', 'Hello Alice!');

// Send file
const file = new File(['content'], 'document.txt', { type: 'text/plain' });
await bitChatService.sendFile('peer_12345', file);
```

### Event Handling

```tsx
// Listen for new messages
bitChatService.on('message-received', (message) => {
  console.log('New message:', message.content);
  // Update UI, show notification, etc.
});

// Listen for connection changes
bitChatService.on('peer-connected', (peerId) => {
  console.log('Peer connected:', peerId);
});
```

## Configuration

### Signaling Server
Configure the signaling server URL for peer discovery:

```bash
# Environment variable
REACT_APP_SIGNALING_SERVER=ws://your-signaling-server.com/signaling
```

### STUN Servers
Default STUN servers are configured for NAT traversal. You can customize them:

```tsx
// In bitChatService.ts
private iceServers: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:your-custom-stun.com:3478' }
];
```

## Security Considerations

### Encryption
- Messages are encrypted using AES-GCM with 256-bit keys
- Each peer pair has a unique encryption key
- Keys are generated locally and not transmitted

### Key Exchange
- Currently uses simple key generation per peer
- For production, implement proper key exchange protocol
- Consider using elliptic curve cryptography for key agreement

### Privacy
- No message content is sent through signaling server
- Local storage contains encrypted message history
- Peer IDs should be treated as public identifiers

## Troubleshooting

### Connection Issues
1. **Firewall/NAT**: Ensure STUN servers are accessible
2. **Signaling Server**: Check WebSocket connection to signaling server
3. **Browser Compatibility**: Requires modern browser with WebRTC support

### Message Delivery
1. **Peer Offline**: Messages only delivered when both peers online
2. **Connection Lost**: Service attempts automatic reconnection
3. **Large Files**: Files are chunked for transmission

### Performance
1. **Many Contacts**: Service scales to hundreds of contacts
2. **Message History**: Local storage has browser limits
3. **File Sharing**: Large files may impact performance

## Development

### Local Testing
1. Open multiple browser tabs/windows
2. Use different peer IDs in each
3. Add each other as contacts
4. Test messaging between tabs

### Production Deployment
1. Set up signaling server for peer discovery
2. Configure STUN/TURN servers for connectivity
3. Implement proper key exchange protocol
4. Add message persistence backend (optional)

## Future Enhancements

- [ ] Voice/video calls over WebRTC
- [ ] Group messaging with multiple peers
- [ ] Message delivery confirmations
- [ ] Offline message queuing
- [ ] Mobile app compatibility
- [ ] Integration with external identity systems
