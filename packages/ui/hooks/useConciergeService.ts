import { useState, useEffect, useCallback, useRef } from 'react';
import { Message, DigitalTwin, ChatState } from '../types';
import { apiService } from '../services/apiService';

export const useConciergeService = (twinId: string, userId: string) => {
  const [chatState, setChatState] = useState<ChatState>({
    messages: [],
    isTyping: false,
    isConnected: false,
    activeTwin: undefined,
  });

  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const MAX_RECONNECT_ATTEMPTS = 5;
  const RECONNECT_DELAY = 3000;

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = apiService.subscribeToConciergeUpdates(twinId, (message: Message) => {
        setChatState(prev => ({
          ...prev,
          messages: [...prev.messages, message],
          isTyping: false,
        }));
      });

      ws.addEventListener('open', () => {
        setChatState(prev => ({ ...prev, isConnected: true }));
        setError(null);
        reconnectAttemptsRef.current = 0;
      });

      ws.addEventListener('close', () => {
        setChatState(prev => ({ ...prev, isConnected: false }));

        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connectWebSocket();
          }, RECONNECT_DELAY * reconnectAttemptsRef.current);
        } else {
          setError('Connection lost. Maximum reconnection attempts exceeded.');
        }
      });

      ws.addEventListener('error', (event) => {
        console.error('WebSocket error:', event);
        setError('Connection error occurred.');
      });

      // Handle typing indicators and other real-time events
      ws.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case 'typing_start':
              setChatState(prev => ({ ...prev, isTyping: true }));
              break;

            case 'typing_stop':
              setChatState(prev => ({ ...prev, isTyping: false }));
              break;

            case 'twin_status_update':
              setChatState(prev => ({
                ...prev,
                activeTwin: { ...prev.activeTwin, ...data.payload } as DigitalTwin
              }));
              break;
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      });

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      setError('Failed to establish real-time connection.');
    }
  }, [twinId]);

  const loadDigitalTwin = useCallback(async () => {
    try {
      const response = await apiService.getDigitalTwin(twinId);
      if (response.success && response.data) {
        setChatState(prev => ({
          ...prev,
          activeTwin: response.data
        }));

        // Load conversation history
        const historyResponse = await apiService.getConversationHistory(twinId, 50);
        if (historyResponse.success && historyResponse.data) {
          setChatState(prev => ({
            ...prev,
            messages: historyResponse.data || []
          }));
        }
      } else {
        setError(response.error || 'Failed to load digital twin');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load digital twin');
    }
  }, [twinId]);

  const sendMessage = useCallback(async (message: Message) => {
    // Optimistically add message to UI
    setChatState(prev => ({
      ...prev,
      messages: [...prev.messages, message],
    }));

    try {
      const response = await apiService.sendConciergeMessage(twinId, message);

      if (!response.success) {
        // Remove optimistic message and show error
        setChatState(prev => ({
          ...prev,
          messages: prev.messages.filter(m => m.id !== message.id),
        }));
        setError(response.error || 'Failed to send message');
        return false;
      }

      return true;
    } catch (err) {
      // Remove optimistic message and show error
      setChatState(prev => ({
        ...prev,
        messages: prev.messages.filter(m => m.id !== message.id),
      }));
      setError(err instanceof Error ? err.message : 'Failed to send message');
      return false;
    }
  }, [twinId]);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    setError(null);
    connectWebSocket();
  }, [connectWebSocket]);

  const clearHistory = useCallback(() => {
    setChatState(prev => ({ ...prev, messages: [] }));
  }, []);

  const updateTwinSettings = useCallback(async (settings: Partial<DigitalTwin>) => {
    try {
      // API call to update twin settings would go here
      setChatState(prev => ({
        ...prev,
        activeTwin: prev.activeTwin ? { ...prev.activeTwin, ...settings } : undefined,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update settings');
    }
  }, []);

  // Initialize connection and load data
  useEffect(() => {
    loadDigitalTwin();
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [loadDigitalTwin, connectWebSocket]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    chatState,
    sendMessage,
    isConnected: chatState.isConnected,
    error,
    reconnect,
    clearHistory,
    updateTwinSettings,
    loadDigitalTwin,
  };
};
