/**
 * ChatWindow Component
 *
 * Main chat interface for P2P messaging with encryption,
 * file sharing, and real-time communication features.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  List,
  ListItem,
  Avatar,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tooltip,
  Snackbar,
  Alert,
  Menu,
  MenuItem,
  Divider,
  Badge
} from '@mui/material';
import {
  Send as SendIcon,
  AttachFile as AttachFileIcon,
  Image as ImageIcon,
  InsertDriveFile as FileIcon,
  Download as DownloadIcon,
  MoreVert as MoreVertIcon,
  Lock as LockIcon,
  LockOpen as LockOpenIcon,
  Circle as CircleIcon,
  Mic as MicIcon,
  MicOff as MicOffIcon
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import { bitChatService, Message, Contact } from '../../services/bitchatService';

interface ChatWindowProps {
  contact: Contact | null;
  onBack?: () => void;
}

interface MessageBubbleProps {
  message: Message;
  isOwn: boolean;
  contact: Contact;
  onDownloadFile?: (message: Message) => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isOwn,
  contact,
  onDownloadFile
}) => {
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleCopyMessage = () => {
    navigator.clipboard.writeText(message.content);
    handleMenuClose();
  };

  const handleDownload = () => {
    if (onDownloadFile) {
      onDownloadFile(message);
    }
    handleMenuClose();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return <ImageIcon />;
    return <FileIcon />;
  };

  return (
    <ListItem
      sx={{
        display: 'flex',
        flexDirection: isOwn ? 'row-reverse' : 'row',
        alignItems: 'flex-start',
        p: 1
      }}
    >
      {!isOwn && (
        <Avatar
          sx={{
            width: 32,
            height: 32,
            mr: 1,
            bgcolor: 'primary.main'
          }}
        >
          {contact.name.charAt(0).toUpperCase()}
        </Avatar>
      )}

      <Box
        sx={{
          maxWidth: '70%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: isOwn ? 'flex-end' : 'flex-start'
        }}
      >
        <Paper
          elevation={1}
          sx={{
            p: 1.5,
            bgcolor: isOwn ? 'primary.main' : 'background.paper',
            color: isOwn ? 'primary.contrastText' : 'text.primary',
            borderRadius: 2,
            position: 'relative',
            '&:hover .message-menu': {
              opacity: 1
            }
          }}
        >
          {message.type === 'file' && message.fileData && (
            <Box sx={{ mb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                {getFileIcon(message.fileData.type)}
                <Box>
                  <Typography variant="body2" fontWeight="bold">
                    {message.fileData.name}
                  </Typography>
                  <Typography variant="caption">
                    {formatFileSize(message.fileData.size)}
                  </Typography>
                </Box>
                <IconButton
                  size="small"
                  onClick={handleDownload}
                  sx={{ ml: 'auto' }}
                >
                  <DownloadIcon />
                </IconButton>
              </Box>
            </Box>
          )}

          {message.type === 'image' && message.fileData && (
            <Box sx={{ mb: 1 }}>
              <img
                src={`data:${message.fileData.type};base64,${message.content}`}
                alt={message.fileData.name}
                style={{
                  maxWidth: '200px',
                  maxHeight: '200px',
                  borderRadius: '8px'
                }}
              />
            </Box>
          )}

          <Typography variant="body2">
            {message.content}
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption" sx={{ opacity: 0.7 }}>
              {formatDistanceToNow(message.timestamp, { addSuffix: true })}
            </Typography>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {message.encrypted && (
                <Tooltip title="End-to-end encrypted">
                  <LockIcon sx={{ fontSize: 12, opacity: 0.7 }} />
                </Tooltip>
              )}

              <IconButton
                size="small"
                className="message-menu"
                onClick={handleMenuOpen}
                sx={{
                  opacity: 0,
                  transition: 'opacity 0.2s',
                  p: 0.25,
                  ml: 0.5
                }}
              >
                <MoreVertIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Box>
          </Box>
        </Paper>

        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={handleMenuClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: isOwn ? 'left' : 'right',
          }}
        >
          <MenuItem onClick={handleCopyMessage}>
            Copy Message
          </MenuItem>
          {message.type === 'file' && (
            <MenuItem onClick={handleDownload}>
              <DownloadIcon sx={{ mr: 1 }} />
              Download File
            </MenuItem>
          )}
        </Menu>
      </Box>
    </ListItem>
  );
};

const ChatWindow: React.FC<ChatWindowProps> = ({ contact, onBack }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [peerTyping, setPeerTyping] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [isConnecting, setIsConnecting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (!contact) return;

    // Load existing messages
    setMessages(bitChatService.getMessages(contact.id));
    setConnectionStatus(bitChatService.getConnectionStatus(contact.id) || 'disconnected');

    // Set up event listeners
    const handleMessageReceived = (message: Message) => {
      if (message.from === contact.id || message.to === contact.id) {
        setMessages(prev => [...prev, message]);
      }
    };

    const handleMessageSent = (message: Message) => {
      if (message.to === contact.id) {
        setMessages(prev => [...prev, message]);
      }
    };

    const handleConnectionStateChanged = (peerId: string, status: string) => {
      if (peerId === contact.id) {
        setConnectionStatus(status);
        setIsConnecting(status === 'connecting');
      }
    };

    const handleTyping = (peerId: string, typing: boolean) => {
      if (peerId === contact.id) {
        setPeerTyping(typing);
      }
    };

    const handlePeerConnected = (peerId: string) => {
      if (peerId === contact.id) {
        setSuccess('Connected to peer');
        setConnectionStatus('connected');
        setIsConnecting(false);
      }
    };

    const handlePeerDisconnected = (peerId: string) => {
      if (peerId === contact.id) {
        setConnectionStatus('disconnected');
        setError('Peer disconnected');
      }
    };

    const handleFileReceived = (fileData: any) => {
      if (fileData.from === contact.id) {
        setSuccess(`Received file: ${fileData.name}`);
      }
    };

    bitChatService.on('message-received', handleMessageReceived);
    bitChatService.on('message-sent', handleMessageSent);
    bitChatService.on('connection-state-changed', handleConnectionStateChanged);
    bitChatService.on('typing', handleTyping);
    bitChatService.on('peer-connected', handlePeerConnected);
    bitChatService.on('peer-disconnected', handlePeerDisconnected);
    bitChatService.on('file-received', handleFileReceived);

    return () => {
      bitChatService.off('message-received', handleMessageReceived);
      bitChatService.off('message-sent', handleMessageSent);
      bitChatService.off('connection-state-changed', handleConnectionStateChanged);
      bitChatService.off('typing', handleTyping);
      bitChatService.off('peer-connected', handlePeerConnected);
      bitChatService.off('peer-disconnected', handlePeerDisconnected);
      bitChatService.off('file-received', handleFileReceived);
    };
  }, [contact]);

  const handleSendMessage = async () => {
    if (!contact || !newMessage.trim() || connectionStatus !== 'connected') return;

    try {
      await bitChatService.sendMessage(contact.id, newMessage.trim());
      setNewMessage('');
      setIsTyping(false);

      // Stop typing indicator
      bitChatService.sendTyping(contact.id, false);
    } catch (error) {
      console.error('Failed to send message:', error);
      setError('Failed to send message');
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNewMessage(event.target.value);

    if (!contact || connectionStatus !== 'connected') return;

    // Send typing indicator
    if (!isTyping && event.target.value.length > 0) {
      setIsTyping(true);
      bitChatService.sendTyping(contact.id, true);
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set new timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      bitChatService.sendTyping(contact.id, false);
    }, 2000);
  };

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !contact || connectionStatus !== 'connected') return;

    try {
      setUploadProgress(0);
      await bitChatService.sendFile(contact.id, file);
      setUploadProgress(null);
      setSuccess(`File sent: ${file.name}`);
    } catch (error) {
      console.error('Failed to send file:', error);
      setError('Failed to send file');
      setUploadProgress(null);
    }

    // Reset file input
    event.target.value = '';
  };

  const handleDownloadFile = (message: Message) => {
    if (!message.fileData?.data) return;

    try {
      const blob = new Blob([message.fileData.data], { type: message.fileData.type });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = message.fileData.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download file:', error);
      setError('Failed to download file');
    }
  };

  const handleConnect = async () => {
    if (!contact) return;

    setIsConnecting(true);
    try {
      await bitChatService.connectToPeer(contact.id);
    } catch (error) {
      console.error('Failed to connect:', error);
      setError('Failed to connect to peer');
      setIsConnecting(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'success';
      case 'connecting': return 'warning';
      case 'disconnected': return 'default';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    const color = getStatusColor(status);
    return (
      <CircleIcon
        sx={{
          fontSize: 12,
          color: color === 'success' ? 'green' :
                 color === 'warning' ? 'orange' :
                 color === 'error' ? 'red' : 'grey'
        }}
      />
    );
  };

  if (!contact) {
    return (
      <Box
        sx={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 2
        }}
      >
        <Typography variant="h6" color="textSecondary">
          Select a contact to start messaging
        </Typography>
        <Typography variant="body2" color="textSecondary" textAlign="center">
          Choose a contact from the list to begin a secure P2P conversation
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Chat Header */}
      <Paper
        elevation={1}
        sx={{
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}
      >
        <Avatar sx={{ bgcolor: 'primary.main' }}>
          {contact.name.charAt(0).toUpperCase()}
        </Avatar>

        <Box sx={{ flex: 1 }}>
          <Typography variant="h6">
            {contact.name}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {getStatusIcon(connectionStatus)}
            <Typography variant="caption" color="textSecondary">
              {connectionStatus === 'connected' ? 'Connected' :
               connectionStatus === 'connecting' ? 'Connecting...' :
               connectionStatus === 'failed' ? 'Connection failed' :
               'Disconnected'}
            </Typography>
            {peerTyping && (
              <Typography variant="caption" color="primary" sx={{ ml: 1 }}>
                typing...
              </Typography>
            )}
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          {connectionStatus !== 'connected' && (
            <Button
              variant="outlined"
              size="small"
              onClick={handleConnect}
              disabled={isConnecting}
            >
              {isConnecting ? 'Connecting...' : 'Connect'}
            </Button>
          )}

          <Chip
            label={connectionStatus === 'connected' ? 'Encrypted' : 'Not connected'}
            icon={connectionStatus === 'connected' ? <LockIcon /> : <LockOpenIcon />}
            color={connectionStatus === 'connected' ? 'success' : 'default'}
            size="small"
          />
        </Box>
      </Paper>

      {/* Connection Progress */}
      {isConnecting && (
        <LinearProgress />
      )}

      {/* Upload Progress */}
      {uploadProgress !== null && (
        <LinearProgress variant="determinate" value={uploadProgress} />
      )}

      {/* Messages Area */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
        {messages.length === 0 ? (
          <Box
            sx={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: 1
            }}
          >
            <Typography variant="body2" color="textSecondary">
              No messages yet
            </Typography>
            <Typography variant="caption" color="textSecondary">
              Send a message to start the conversation
            </Typography>
          </Box>
        ) : (
          <List sx={{ p: 0 }}>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isOwn={message.from === bitChatService.getLocalPeerId()}
                contact={contact}
                onDownloadFile={handleDownloadFile}
              />
            ))}
          </List>
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Message Input */}
      <Paper
        elevation={1}
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: 'divider',
          display: 'flex',
          gap: 1,
          alignItems: 'flex-end'
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          hidden
          onChange={handleFileUpload}
        />

        <IconButton
          onClick={handleFileSelect}
          disabled={connectionStatus !== 'connected'}
          color="primary"
        >
          <AttachFileIcon />
        </IconButton>

        <TextField
          ref={messageInputRef}
          fullWidth
          multiline
          maxRows={4}
          placeholder={
            connectionStatus === 'connected'
              ? "Type a message..."
              : "Connect to start messaging"
          }
          value={newMessage}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          disabled={connectionStatus !== 'connected'}
          variant="outlined"
          size="small"
        />

        <IconButton
          onClick={handleSendMessage}
          disabled={!newMessage.trim() || connectionStatus !== 'connected'}
          color="primary"
        >
          <SendIcon />
        </IconButton>
      </Paper>

      {/* Notifications */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!success}
        autoHideDuration={4000}
        onClose={() => setSuccess(null)}
      >
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ChatWindow;
