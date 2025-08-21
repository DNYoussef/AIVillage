import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import BitChatInterface from './BitChatInterface';

// Mock the BitChat service
const mockBitChatService = {
  isConnected: true,
  peers: [
    { id: 'peer1', name: 'Alice', status: 'online' },
    { id: 'peer2', name: 'Bob', status: 'online' }
  ],
  messages: [
    { id: 'msg1', from: 'peer1', content: 'Hello there!', timestamp: new Date().toISOString() }
  ],
  sendMessage: jest.fn(),
  connectToPeer: jest.fn()
};

jest.mock('../../hooks/useBitChatService', () => ({
  useBitChatService: () => mockBitChatService
}));

describe('BitChatInterface', () => {
  test('renders BitChat interface', () => {
    render(<BitChatInterface />);

    expect(screen.getByText('BitChat P2P Messaging')).toBeInTheDocument();
    expect(screen.getByText('Network Status')).toBeInTheDocument();
    expect(screen.getByText('Connected Peers')).toBeInTheDocument();
  });

  test('displays connected peers', () => {
    render(<BitChatInterface />);

    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();
  });

  test('displays chat messages', () => {
    render(<BitChatInterface />);

    expect(screen.getByText('Hello there!')).toBeInTheDocument();
  });

  test('allows sending messages', () => {
    render(<BitChatInterface />);

    const messageInput = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(messageInput, { target: { value: 'Test message' } });
    fireEvent.click(sendButton);

    expect(mockBitChatService.sendMessage).toHaveBeenCalledWith('Test message');
  });

  test('shows connection status', () => {
    render(<BitChatInterface />);

    expect(screen.getByText('Connected')).toBeInTheDocument();
  });
});
