# AIVillage Web Client

A modern, accessible web interface for the AIVillage Distributed AI Platform built with Vue 3, TypeScript, and comprehensive accessibility features.

## Features

### 🤖 AI Integration
- **Multi-Agent Chat**: Interact with specialized AI agents (King, Magi, Sage, Oracle, Navigator)
- **Advanced RAG Queries**: Retrieval-augmented generation with Bayesian trust networks
- **Agent Management**: Monitor and assign tasks to AI agents
- **P2P Network Status**: Real-time peer-to-peer network monitoring
- **Digital Twin Assistant**: Privacy-preserving personal AI assistant

### ⚡ Performance & Caching
- **Intelligent Caching**: Multi-layer caching with configurable TTL
- **Service Worker**: PWA functionality with offline support
- **API Response Caching**: Reduces redundant API calls
- **Request Deduplication**: Automatic deduplication of identical requests
- **Performance Monitoring**: Built-in performance metrics and analytics

### ♿ Accessibility First
- **WCAG 2.1 AA Compliant**: Meets international accessibility standards
- **Screen Reader Support**: Full ARIA labeling and live regions
- **Keyboard Navigation**: Complete keyboard accessibility
- **Focus Management**: Proper focus indicators and management
- **Reduced Motion Support**: Respects `prefers-reduced-motion`
- **High Contrast Mode**: Enhanced visibility for users with visual impairments
- **Large Text Mode**: Scalable text for better readability
- **Skip Links**: Quick navigation for screen reader users

### 🌐 Internationalization
- **Multi-Language Support**: English, Spanish, French, German, Japanese, Chinese
- **RTL Support**: Right-to-left language support (planned)
- **Locale-Aware Formatting**: Dates, numbers, and currencies

### 🔒 Security & Privacy
- **Content Security Policy**: Strict CSP headers
- **XSS Protection**: Built-in cross-site scripting protection
- **HTTPS Only**: Secure communication with all APIs
- **Privacy Controls**: User-configurable privacy settings

## Quick Start

### Prerequisites
- Node.js 16+
- npm or pnpm
- Modern web browser with ES2020 support

### Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage/clients/web

# Install dependencies
npm install

# Start development server
npm run dev
```

### Environment Variables

Create a `.env` file in the web client directory:

```bash
# API Configuration
VITE_API_BASE_URL=https://api.aivillage.io/v1

# Development settings
VITE_DEV_MODE=true
VITE_ENABLE_ANALYTICS=false
```

## Usage

### Basic Chat

```typescript
import { useApiStore } from '@/stores/api'

const apiStore = useApiStore()

// Initialize with API key
apiStore.initializeClient('your-api-key')

// Send a chat message
const response = await apiStore.sendChatMessage('Hello, how can you help me?', {
  agentPreference: 'magi',
  mode: 'balanced'
})
```

### RAG Queries

```typescript
// Process a knowledge query
const result = await apiStore.processQuery('What are the best practices for AI safety?', {
  mode: 'comprehensive',
  includeSources: true,
  maxResults: 10
})
```

### Caching

```typescript
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

// Manual cache management
appStore.setCache('my-key', data, 60000) // 1 minute TTL
const cached = appStore.getCache('my-key')

// Clear specific cache pattern
appStore.clearCache(/^api-/)

// Cache statistics
console.log('Hit rate:', appStore.cacheHitRate)
```

## Architecture

### Component Structure
```
src/
├── components/          # Reusable UI components
│   ├── AccessibilityMenu.vue
│   ├── NetworkStatusIndicator.vue
│   └── ...
├── stores/             # Pinia state management
│   ├── app.ts         # App-wide state & caching
│   └── api.ts         # API client & request handling
├── views/             # Page components
│   ├── ChatView.vue
│   ├── QueryView.vue
│   └── ...
├── router/            # Vue Router configuration
├── locales/           # i18n translation files
└── style.css          # Global styles & accessibility
```

### State Management

The application uses **Pinia** for state management with two main stores:

#### App Store (`useAppStore`)
- Theme and language preferences
- Accessibility settings
- Caching layer with TTL support
- Error handling
- Network status

#### API Store (`useApiStore`)
- AIVillage API client wrapper
- Request retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Idempotency key management
- Rate limiting awareness

### Caching Strategy

**Multi-Layer Caching:**

1. **Memory Cache** (App Store)
   - TTL-based expiration
   - Pattern-based clearing
   - Hit/miss statistics

2. **Service Worker Cache** (PWA)
   - API response caching
   - Static asset caching
   - Offline fallback support

3. **Browser Cache** (HTTP)
   - Leverages HTTP caching headers
   - Automatic cache validation

**Cache Invalidation:**
- Time-based expiration (TTL)
- Manual clearing by pattern
- Network-first for critical data
- Cache-first for static assets

## Accessibility Features

### WCAG 2.1 AA Compliance

✅ **Perceivable**
- Color contrast ratios meet 4.5:1 minimum
- Text scaling up to 200% without horizontal scrolling
- Alt text for all images
- Captions for video content (when applicable)

✅ **Operable**
- Full keyboard navigation
- No seizure-inducing content
- Sufficient time limits with extensions
- Skip links for navigation

✅ **Understandable**
- Consistent navigation and layout
- Clear error messages and form validation
- Language identification
- Predictable user interface changes

✅ **Robust**
- Valid semantic HTML
- ARIA labels and roles
- Compatible with assistive technologies
- Progressive enhancement

### Accessibility Settings

Users can customize:
- **Reduced Motion**: Minimizes animations and transitions
- **High Contrast**: Increases color contrast for better visibility
- **Large Text**: Increases text size for better readability
- **Screen Reader Mode**: Optimizes interface for screen readers
- **Focus Indicators**: Customizable focus outline styles

### Screen Reader Support

- Complete ARIA labeling
- Live regions for dynamic content
- Form associations and descriptions
- Semantic heading structure
- Table headers and captions

## Performance

### Optimization Techniques

**Code Splitting:**
- Route-based code splitting
- Dynamic imports for components
- Vendor chunk separation

**Asset Optimization:**
- Image lazy loading
- Font display optimization
- Critical CSS inlining
- Resource preloading

**Bundle Analysis:**
```bash
npm run build
npm run analyze
```

**Performance Budgets:**
- JavaScript: < 250KB gzipped
- CSS: < 50KB gzipped
- Images: WebP with fallbacks
- Fonts: Subset and preload

### PWA Features

- **Service Worker**: Caching and offline support
- **Web App Manifest**: Native app-like experience
- **Push Notifications**: Background updates (planned)
- **Background Sync**: Offline action queuing (planned)

## Testing

```bash
# Run all tests
npm run test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# E2E tests (Playwright)
npm run test:e2e
```

### Test Coverage Goals
- Unit tests: > 80% coverage
- Integration tests: Critical user flows
- Accessibility tests: Automated a11y testing
- Performance tests: Core Web Vitals monitoring

## Deployment

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment Configuration

**Production Environment Variables:**
```bash
VITE_API_BASE_URL=https://api.aivillage.io/v1
VITE_ENABLE_ANALYTICS=true
VITE_SENTRY_DSN=your-sentry-dsn
```

## Browser Support

### Supported Browsers

✅ **Modern Browsers** (ES2020+ support):
- Chrome 88+
- Firefox 87+
- Safari 14+
- Edge 88+

⚠️ **Limited Support**:
- IE 11: Not supported (recommend modern browser message)
- Older mobile browsers: Basic functionality only

### Feature Detection

The application uses progressive enhancement:
```javascript
// Service Worker support
if ('serviceWorker' in navigator) {
  // Register SW
}

// Web Share API
if (navigator.share) {
  // Enable native sharing
}
```

## Contributing

### Development Setup

1. **Clone and install dependencies**
2. **Start development server**: `npm run dev`
3. **Run tests**: `npm run test`
4. **Check accessibility**: Use browser dev tools and screen reader testing

### Code Quality

**Pre-commit Hooks:**
- ESLint with Vue and accessibility rules
- Prettier formatting
- TypeScript type checking
- Accessibility linting (eslint-plugin-vuejs-accessibility)

**Coding Standards:**
- Vue 3 Composition API
- TypeScript strict mode
- Semantic HTML
- BEM CSS methodology (within Tailwind)

### Accessibility Testing

**Manual Testing:**
1. Keyboard-only navigation
2. Screen reader testing (NVDA, JAWS, VoiceOver)
3. High contrast mode testing
4. Reduced motion preference testing

**Automated Testing:**
```bash
# Run accessibility tests
npm run test:a11y

# Lighthouse accessibility audit
npm run audit:a11y
```

## Troubleshooting

### Common Issues

**Build Errors:**
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

**API Connection Issues:**
- Check `VITE_API_BASE_URL` environment variable
- Verify CORS settings on API server
- Check browser network tab for failed requests

**Accessibility Issues:**
- Test with keyboard navigation
- Use browser accessibility tools
- Validate HTML with W3C validator
- Test with actual screen readers

### Performance Issues

**Bundle Size:**
```bash
# Analyze bundle size
npm run build
npm run analyze
```

**Runtime Performance:**
- Use React DevTools Profiler
- Monitor Core Web Vitals
- Check for memory leaks
- Optimize re-renders

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **Issues**: [GitHub Issues](https://github.com/DNYoussef/AIVillage/issues)
- **Accessibility**: [accessibility@aivillage.io](mailto:accessibility@aivillage.io)
- **Security**: [security@aivillage.io](mailto:security@aivillage.io)

## License

MIT License - see [LICENSE](../../LICENSE) file for details.
