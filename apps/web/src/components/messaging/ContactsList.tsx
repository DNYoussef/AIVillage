/**
 * ContactsList Component
 *
 * Manages P2P contacts for BitChat messaging including discovery,
 * connection status, and contact management.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Avatar,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Chip,
  Typography,
  Menu,
  MenuItem,
  Badge,
  Tooltip,
  Card,
  CardContent,
  InputAdornment,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  PersonAdd as PersonAddIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Circle as CircleIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  QrCode as QrCodeIcon,
  Share as ShareIcon
} from '@mui/icons-material';
import { bitChatService, Contact } from '../../services/bitchatService';

interface ContactsListProps {
  selectedContact: string | null;
  onContactSelect: (contactId: string) => void;
  onNewMessage: (contact: Contact) => void;
}

interface AddContactDialogProps {
  open: boolean;
  onClose: () => void;
  onAdd: (contact: Omit<Contact, 'lastSeen' | 'status'>) => void;
}

const AddContactDialog: React.FC<AddContactDialogProps> = ({ open, onClose, onAdd }) => {
  const [name, setName] = useState('');
  const [peerId, setPeerId] = useState('');
  const [publicKey, setPublicKey] = useState('');
  const [error, setError] = useState('');

  const handleAdd = () => {
    if (!name.trim() || !peerId.trim()) {
      setError('Name and Peer ID are required');
      return;
    }

    try {
      onAdd({
        id: peerId.trim(),
        name: name.trim(),
        publicKey: publicKey.trim() || peerId.trim() // Use peerId as fallback
      });

      // Reset form
      setName('');
      setPeerId('');
      setPublicKey('');
      setError('');
      onClose();
    } catch (err) {
      setError('Failed to add contact');
    }
  };

  const handleClose = () => {
    setName('');
    setPeerId('');
    setPublicKey('');
    setError('');
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Add New Contact</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          <TextField
            label="Contact Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            fullWidth
            required
            helperText="Display name for this contact"
          />
          <TextField
            label="Peer ID"
            value={peerId}
            onChange={(e) => setPeerId(e.target.value)}
            fullWidth
            required
            helperText="Unique identifier for P2P connection"
          />
          <TextField
            label="Public Key (Optional)"
            value={publicKey}
            onChange={(e) => setPublicKey(e.target.value)}
            fullWidth
            multiline
            rows={3}
            helperText="Public key for end-to-end encryption"
          />
          {error && (
            <Typography color="error" variant="body2">
              {error}
            </Typography>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button onClick={handleAdd} variant="contained">
          Add Contact
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const ContactsList: React.FC<ContactsListProps> = ({
  selectedContact,
  onContactSelect,
  onNewMessage
}) => {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [filteredContacts, setFilteredContacts] = useState<Contact[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedContactForMenu, setSelectedContactForMenu] = useState<string | null>(null);
  const [discoveredPeers, setDiscoveredPeers] = useState<string[]>([]);
  const [connectionStatuses, setConnectionStatuses] = useState<Map<string, string>>(new Map());

  useEffect(() => {
    // Load initial contacts
    setContacts(bitChatService.getContacts());

    // Set up event listeners
    const handleContactAdded = (contact: Contact) => {
      setContacts(prev => [...prev, contact]);
    };

    const handleContactRemoved = (contactId: string) => {
      setContacts(prev => prev.filter(c => c.id !== contactId));
    };

    const handleContactUpdated = (contact: Contact) => {
      setContacts(prev => prev.map(c => c.id === contact.id ? contact : c));
    };

    const handlePeersDiscovered = (peers: string[]) => {
      setDiscoveredPeers(peers);
    };

    const handleConnectionStateChanged = (peerId: string, status: string) => {
      setConnectionStatuses(prev => new Map(prev.set(peerId, status)));
    };

    bitChatService.on('contact-added', handleContactAdded);
    bitChatService.on('contact-removed', handleContactRemoved);
    bitChatService.on('contact-updated', handleContactUpdated);
    bitChatService.on('peers-discovered', handlePeersDiscovered);
    bitChatService.on('connection-state-changed', handleConnectionStateChanged);

    return () => {
      bitChatService.off('contact-added', handleContactAdded);
      bitChatService.off('contact-removed', handleContactRemoved);
      bitChatService.off('contact-updated', handleContactUpdated);
      bitChatService.off('peers-discovered', handlePeersDiscovered);
      bitChatService.off('connection-state-changed', handleConnectionStateChanged);
    };
  }, []);

  useEffect(() => {
    // Filter contacts based on search term
    const filtered = contacts.filter(contact =>
      contact.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      contact.id.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredContacts(filtered);
  }, [contacts, searchTerm]);

  const handleAddContact = (contactData: Omit<Contact, 'lastSeen' | 'status'>) => {
    bitChatService.addContact(contactData);
  };

  const handleContactClick = (contact: Contact) => {
    onContactSelect(contact.id);

    // Attempt to connect if not already connected
    const status = bitChatService.getConnectionStatus(contact.id);
    if (!status || status === 'disconnected' || status === 'failed') {
      bitChatService.connectToPeer(contact.id).catch(error => {
        console.error('Failed to connect to peer:', error);
      });
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, contactId: string) => {
    event.stopPropagation();
    setMenuAnchor(event.currentTarget);
    setSelectedContactForMenu(contactId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedContactForMenu(null);
  };

  const handleRemoveContact = () => {
    if (selectedContactForMenu) {
      bitChatService.removeContact(selectedContactForMenu);
    }
    handleMenuClose();
  };

  const handleConnectToPeer = () => {
    if (selectedContactForMenu) {
      bitChatService.connectToPeer(selectedContactForMenu).catch(error => {
        console.error('Failed to connect to peer:', error);
      });
    }
    handleMenuClose();
  };

  const handleDisconnectPeer = () => {
    if (selectedContactForMenu) {
      bitChatService.disconnectPeer(selectedContactForMenu);
    }
    handleMenuClose();
  };

  const getStatusColor = (status: Contact['status']) => {
    switch (status) {
      case 'online': return 'success';
      case 'connecting': return 'warning';
      case 'offline': return 'default';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: Contact['status']) => {
    const color = getStatusColor(status);
    return (
      <CircleIcon
        sx={{
          fontSize: 12,
          color: color === 'success' ? 'green' :
                 color === 'warning' ? 'orange' : 'grey'
        }}
      />
    );
  };

  const getConnectionStatus = (contactId: string): string => {
    return connectionStatuses.get(contactId) || 'disconnected';
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Contacts</Typography>
          <Box>
            <Tooltip title="Refresh peer discovery">
              <IconButton size="small" onClick={() => window.location.reload()}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Add contact">
              <IconButton
                size="small"
                onClick={() => setAddDialogOpen(true)}
                color="primary"
              >
                <AddIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <TextField
          fullWidth
          size="small"
          placeholder="Search contacts..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
      </Box>

      {/* Discovered Peers */}
      {discoveredPeers.length > 0 && (
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2" color="textSecondary" gutterBottom>
            Discovered Peers
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {discoveredPeers.map(peerId => (
              <Chip
                key={peerId}
                label={peerId}
                size="small"
                onClick={() => {
                  // Quick add discovered peer
                  handleAddContact({
                    id: peerId,
                    name: `Peer ${peerId.slice(-6)}`,
                    publicKey: peerId
                  });
                }}
                icon={<PersonAddIcon />}
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Contacts List */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {filteredContacts.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography color="textSecondary">
              {searchTerm ? 'No contacts found' : 'No contacts yet'}
            </Typography>
            <Button
              startIcon={<AddIcon />}
              onClick={() => setAddDialogOpen(true)}
              sx={{ mt: 1 }}
            >
              Add your first contact
            </Button>
          </Box>
        ) : (
          <List>
            {filteredContacts.map((contact) => {
              const connectionStatus = getConnectionStatus(contact.id);
              const isSelected = selectedContact === contact.id;

              return (
                <ListItem
                  key={contact.id}
                  button
                  selected={isSelected}
                  onClick={() => handleContactClick(contact)}
                  sx={{
                    borderBottom: 1,
                    borderColor: 'divider',
                    '&:hover': {
                      bgcolor: 'action.hover'
                    }
                  }}
                >
                  <ListItemIcon>
                    <Badge
                      overlap="circular"
                      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                      badgeContent={getStatusIcon(contact.status)}
                    >
                      <Avatar sx={{ bgcolor: 'primary.main' }}>
                        {contact.name.charAt(0).toUpperCase()}
                      </Avatar>
                    </Badge>
                  </ListItemIcon>

                  <ListItemText
                    primary={contact.name}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {contact.id}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                          <Chip
                            label={contact.status}
                            size="small"
                            color={getStatusColor(contact.status)}
                            variant="outlined"
                          />
                          {connectionStatus !== 'disconnected' && (
                            <Chip
                              label={connectionStatus}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Box>
                      </Box>
                    }
                  />

                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={(e) => handleMenuOpen(e, contact.id)}
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              );
            })}
          </List>
        )}
      </Box>

      {/* Local Peer Info */}
      <Card sx={{ m: 2, mt: 0 }}>
        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
          <Typography variant="subtitle2" gutterBottom>
            Your Peer ID
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
            {bitChatService.getLocalPeerId()}
          </Typography>
          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
            <Tooltip title="Share QR Code">
              <IconButton size="small">
                <QrCodeIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Copy to clipboard">
              <IconButton
                size="small"
                onClick={() => {
                  navigator.clipboard.writeText(bitChatService.getLocalPeerId());
                }}
              >
                <ShareIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </CardContent>
      </Card>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleConnectToPeer}>
          <PersonAddIcon sx={{ mr: 1 }} />
          Connect
        </MenuItem>
        <MenuItem onClick={handleDisconnectPeer}>
          <CircleIcon sx={{ mr: 1 }} />
          Disconnect
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleMenuClose}>
          <EditIcon sx={{ mr: 1 }} />
          Edit Contact
        </MenuItem>
        <MenuItem onClick={handleRemoveContact} sx={{ color: 'error.main' }}>
          <DeleteIcon sx={{ mr: 1 }} />
          Remove Contact
        </MenuItem>
      </Menu>

      {/* Add Contact Dialog */}
      <AddContactDialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        onAdd={handleAddContact}
      />
    </Box>
  );
};

export default ContactsList;
