# AIVillage Voice Interface Documentation

## Overview

The AIVillage Consumer UI now features a comprehensive voice recognition and speech synthesis system that enables natural voice conversations with AI agents. This implementation provides a seamless, accessible, and secure voice interaction experience.

## Features Implemented

### 🎙️ Web Speech API Integration
- **Real-time Voice Recognition**: Continuous speech recognition with interim results
- **Multi-language Support**: Configurable language detection (default: en-US)
- **Error Handling**: Comprehensive error recovery and user feedback
- **Permission Management**: Graceful microphone permission requests

### 🔊 Speech Synthesis
- **Agent-Specific Voices**: Different voice characteristics for each AI agent
  - **Concierge**: Female voice with professional tone
  - **Professor**: Male voice with educational clarity
  - **Steward**: Female voice with warm, nurturing quality
- **Configurable Speech Parameters**: Adjustable rate, pitch, and volume
- **Response Timing**: Natural pauses and speech rhythm

### 📊 Audio Visualization
- **Real-time Audio Levels**: Visual feedback during voice input
- **Dynamic Sphere Animation**: Voice sphere responds to audio intensity
- **State Indicators**: Visual cues for listening, processing, and speaking states
- **Responsive Design**: Scales appropriately across devices

### 🤖 Multi-Agent Support
- **Three Specialized Agents**:
  - **Concierge**: Personal AI assistant for general queries
  - **Professor**: Educational AI tutor for learning support
  - **Steward**: Horticulture and gardening expert
- **Context-Aware Responses**: Agents provide specialized responses based on their domain
- **Session Management**: Maintains conversation context across interactions

### 🔒 Security & Privacy
- **Secure Permissions**: Explicit microphone access requests
- **Local Processing**: Speech recognition happens locally when possible
- **Graceful Degradation**: Automatic fallback to text mode when voice unavailable
- **No Audio Storage**: Voice data is processed in real-time, not stored

## Technical Implementation

### Architecture Components

```typescript
// Core Voice Recognition Interface
interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

// Chat Response Structure
interface ChatResponse {
  response: string;
  agent: string;
  timestamp: string;
}
```

### Key Files Modified

1. **`App.tsx`** - Main application component with voice integration
2. **`App.css`** - Enhanced styles for voice interface elements
3. **`services/apiService.ts`** - API integration for chat functionality
4. **`demo-voice.html`** - Feature demonstration page

### State Management

```typescript
// Voice Interface States
const [isListening, setIsListening] = useState(false);
const [isProcessing, setIsProcessing] = useState(false);
const [isSpeaking, setIsSpeaking] = useState(false);
const [hasPermission, setHasPermission] = useState<boolean | null>(null);
const [transcript, setTranscript] = useState('');
const [audioLevel, setAudioLevel] = useState(0);
const [chatHistory, setChatHistory] = useState([]);
```

## User Interface Elements

### Voice Sphere
- **Interactive Design**: Central voice interaction point
- **Visual States**: Different animations for idle, listening, processing, speaking
- **Audio Visualization**: Real-time audio level bars during recording
- **Click Controls**: Tap to start/stop voice interaction

### Status Indicators
- **Listening**: "🎤 Listening... (live transcript)"
- **Processing**: "🤔 Processing..."
- **Speaking**: "🗣️ Speaking..."
- **Ready**: "Click to talk to [Agent Name]"
- **Error**: Warning banner with error details

### Chat History
- **Voice Mode**: Compact conversation history with timestamps
- **Text Mode**: Full chat interface with typing indicators
- **Message Types**: Visually distinct user and AI messages
- **Timestamps**: Local time display for each interaction

## Browser Compatibility

### Fully Supported
- ✅ **Chrome 25+** - Full Web Speech API support
- ✅ **Firefox 94+** - Complete implementation
- ✅ **Safari 14.1+** - iOS and macOS support
- ✅ **Edge 79+** - Chromium-based full support

### Partial Support
- ⚠️ **Opera** - Limited Speech Recognition support
- ⚠️ **Mobile Browsers** - May require user interaction to start

### Not Supported
- ❌ **Internet Explorer** - No Web Speech API support
- ❌ **Older Browsers** - Automatic fallback to text mode

## API Integration

### Chat Endpoint

```typescript
// Real API Call
POST /api/chat
{
  "message": string,
  "agent": "concierge" | "professor" | "steward",
  "sessionId": string
}

// Response
{
  "response": string,
  "agent": string,
  "timestamp": string
}
```

### Mock Implementation
For development, the system includes intelligent mock responses:

```typescript
const responses = {
  concierge: [
    "Hello! I'm your personal AI concierge. How can I assist you today?",
    "I'm here to help you navigate the AIVillage ecosystem...",
    // ... more responses
  ],
  professor: [
    "As your AI tutor, I'm excited to help you learn!",
    "Learning is a wonderful journey...",
    // ... more responses
  ],
  steward: [
    "Welcome to your garden companion!",
    "Every plant has its own story...",
    // ... more responses
  ]
};
```

## Accessibility Features

### Visual Accessibility
- **High Contrast Mode**: Improved visibility in high contrast environments
- **Reduced Motion**: Respects user's motion preferences
- **Focus Indicators**: Clear keyboard navigation support
- **Screen Reader Support**: Proper ARIA labels and semantic HTML

### Audio Accessibility
- **Volume Controls**: Adjustable speech synthesis volume
- **Speed Controls**: Configurable speech rate
- **Text Fallback**: Always available as alternative input method
- **Visual Transcription**: Live transcript display during voice input

## Performance Optimizations

### Memory Management
- **Cleanup Handlers**: Proper disposal of audio contexts and streams
- **Event Listeners**: Automatic cleanup on component unmount
- **Stream Management**: Efficient microphone stream handling

### Network Efficiency
- **Debounced Requests**: Prevents rapid API calls
- **Connection Pooling**: Reuses API connections when possible
- **Fallback Responses**: Local mock responses when API unavailable

### Battery Optimization
- **Conditional Processing**: Audio analysis only when actively listening
- **Smart Timeouts**: Automatic session cleanup
- **Efficient Animations**: CSS-based animations with reduced computational overhead

## Error Handling

### Permission Errors
```typescript
// Microphone permission denied
setError('Microphone access denied. Please enable microphone permissions.');
setHasPermission(false);
```

### Recognition Errors
```typescript
// Speech recognition failures
recognition.onerror = (event) => {
  setError(`Speech recognition error: ${event.error}`);
  setIsListening(false);
};
```

### API Errors
```typescript
// Network or API failures
catch (error) {
  const errorMessage = 'Sorry, I encountered an error. Please try again.';
  await speakResponse(errorMessage);
}
```

## Development Setup

### Prerequisites
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Environment Variables
```bash
# API Configuration (optional for mock mode)
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

### Testing Voice Features
1. **Open Development Server**: Navigate to `http://localhost:5173`
2. **Grant Microphone Permission**: Click "Enable Microphone" when prompted
3. **Select AI Agent**: Choose Concierge, Professor, or Steward
4. **Start Voice Interaction**: Click the voice sphere to begin speaking
5. **Test Fallback**: Switch to text mode to verify both interfaces work

## Production Deployment

### HTTPS Requirement
Voice features require HTTPS in production:
```nginx
# Nginx configuration example
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Security Headers
```javascript
// Required security headers
{
  "Permissions-Policy": "microphone=(self)",
  "Content-Security-Policy": "default-src 'self'; microphone 'self'",
  "Feature-Policy": "microphone 'self'"
}
```

## Future Enhancements

### Planned Features
- **Voice Commands**: Shortcut phrases for common actions
- **Multiple Languages**: Expanded language support beyond English
- **Voice Training**: Personalized voice recognition adaptation
- **Noise Cancellation**: Advanced audio filtering capabilities
- **Offline Mode**: Local speech processing for privacy-sensitive environments

### Integration Opportunities
- **WebRTC**: Real-time communication with other users
- **AI Voice Cloning**: Custom voice synthesis for agents
- **Emotion Detection**: Sentiment analysis from voice tone
- **Background Conversation**: Ambient listening mode

## Support and Troubleshooting

### Common Issues

1. **No Voice Recognition**
   - Check browser compatibility
   - Verify microphone permissions
   - Test with different microphone devices
   - Ensure HTTPS in production

2. **Poor Audio Quality**
   - Check microphone positioning
   - Reduce background noise
   - Verify browser audio settings
   - Test different browsers

3. **API Connection Issues**
   - Verify network connectivity
   - Check API endpoint configuration
   - Review CORS settings
   - Monitor server logs

### Debug Mode
Enable debug logging:
```typescript
// Add to development environment
localStorage.setItem('voice-debug', 'true');
```

## Contributing

### Code Style
- Follow existing TypeScript patterns
- Maintain accessibility standards
- Add comprehensive error handling
- Include unit tests for new features

### Testing Voice Features
- Test across multiple browsers
- Verify on different devices
- Check accessibility compliance
- Validate error scenarios

---

For additional support or feature requests, please refer to the main AIVillage documentation or create an issue in the project repository.
