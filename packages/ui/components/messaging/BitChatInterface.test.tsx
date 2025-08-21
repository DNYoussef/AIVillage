import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import BitChatInterface from './BitChatInterface';

// Mock the BitChat service
const mockBitChatService = {
  messagingState: {
    peers: [
      { id: 'peer1', name: 'Alice', status: 'online', lastSeen: new Date(), publicKey: 'key1' },
      { id: 'peer2', name: 'Bob', status: 'online', lastSeen: new Date(), publicKey: 'key2' }
    ],
    conversations: {
      peer1: [
        { 
          id: 'msg1', 
          sender: 'peer1', 
          content: 'Hello there!', 
          timestamp: new Date(),
          type: 'user',
          encrypted: true,
          recipient: 'test-user',
          deliveryStatus: 'delivered'
        }
      ]
    },
    activeChat: null,
    isDiscovering: false
  },
  sendMessage: jest.fn(),
  discoverPeers: jest.fn(),
  connectToPeer: jest.fn(),
  disconnectFromPeer: jest.fn(),
  meshStatus: {
    health: 'good',
    connectivity: 95,
    throughput: 1024
  },
  encryptionStatus: {
    enabled: true,
    protocol: 'ChaCha20-Poly1305'
  }
};

jest.mock('../../hooks/useBitChatService', () => ({
  useBitChatService: () => mockBitChatService
}));

describe('BitChatInterface', () => {
  const defaultProps = {
    userId: 'test-user',
    onPeerConnect: jest.fn(),
    onMessageReceived: jest.fn()
  };

  test('renders BitChat interface', () => {
    render(<BitChatInterface {...defaultProps} />);

    expect(screen.getByText('BitChat Mesh Network')).toBeInTheDocument();
    expect(screen.getByText('Network Status')).toBeInTheDocument();
    expect(screen.getByText('Connected Peers')).toBeInTheDocument();
  });

  test('displays connected peers', () => {
    render(<BitChatInterface {...defaultProps} />);

    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();
  });

  test('displays chat messages when peer selected', () => {
    render(<BitChatInterface {...defaultProps} />);

    // First select a peer to see messages
    fireEvent.click(screen.getByText('Alice'));
    expect(screen.getByText('Hello there!')).toBeInTheDocument();
  });

  test('allows sending messages', () => {
    render(<BitChatInterface {...defaultProps} />);

    // First select a peer
    fireEvent.click(screen.getByText('Alice'));
    
    const messageInput = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(messageInput, { target: { value: 'Test message' } });
    fireEvent.click(sendButton);

    expect(mockBitChatService.sendMessage).toHaveBeenCalled();
  });

  test('shows network health status', () => {
    render(<BitChatInterface {...defaultProps} />);

    // Should show network status information
    expect(screen.getByText('Network Status')).toBeInTheDocument();
  });
});
