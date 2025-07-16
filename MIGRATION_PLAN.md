# Expo Monorepo Migration Plan

## Current State Analysis

### UI Codebase (`/ui`)
- **Type**: Simple HTML/CSS/JS dashboard
- **Components**:
  - `ConfidenceChip.jsx` - Displays confidence levels with color coding
  - `EvidencePanel.jsx` - Shows evidence with confidence chips
- **Functionality**: Basic dashboard with chat, upload, bayes graph, logs, status
- **Styling**: Tailwind CSS classes

### Mobile App Codebase (`/mobile-app`)
- **Type**: React Native app (appears to be Expo-based)
- **Components**:
  - `EvidencePanel.js` - Similar to UI version but for RN
  - `Flashcard.js` - Animated flashcard component
  - `LottieSplash.js` - Splash screen with Lottie animation
  - `ProgressBar.js` - Progress indicator
  - `TranscriptBubble.js` - Chat bubble component
  - `WaveformRecorder.js` - Audio recording component
- **Screens**: HomeScreen, TwinScreen, LearnScreen
- **Navigation**: Tab-based navigation with icons
- **Features**: i18n support, API integration

## Duplicate Components Identified

1. **EvidencePanel** - Both codebases have similar components for displaying evidence
2. **Confidence-related UI** - Both need confidence indicators

## Target Monorepo Structure

```
apps/
├── web/                    # Next.js + Expo Router web app
├── mobile/                 # Expo mobile app
packages/
├── ui-kit/                 # Shared UI components
├── shared/                 # Shared utilities, types, constants
├── api-client/             # API client logic
└── config/                 # Shared configuration (eslint, typescript, etc.)
```

## Migration Strategy

### Phase 1: Foundation
1. Create Turborepo with TypeScript
2. Setup shared configurations
3. Create base packages structure

### Phase 2: UI Kit Development
1. Extract common components to `packages/ui-kit`
2. Create platform-agnostic components using React Native Web
3. Implement design system with consistent theming

### Phase 3: App Migration
1. Setup web app with Next.js + Expo Router
2. Setup mobile app with Expo
3. Migrate screens incrementally

### Phase 4: Integration
1. Setup shared API client
2. Implement shared state management
3. Configure CI/CD pipeline

## Package Dependencies Strategy

### Core Dependencies
- `expo` - Universal platform
- `expo-router` - File-based routing
- `react-native-web` - Web compatibility
- `next` - Web framework
- `typescript` - Type safety
- `tailwindcss` - Styling (web)
- `nativewind` - Styling (mobile)

### Development Dependencies
- `turbo` - Build system
- `eslint` - Linting
- `prettier` - Code formatting
- `@expo/metro-config` - Metro bundler config

## Import Strategy

```typescript
// From ui-kit
import { Button, ConfidenceChip, EvidencePanel } from '@aivillage/ui-kit'

// From shared
import { apiClient } from '@aivillage/api-client'
import { colors, spacing } from '@aivillage/shared'

// From config
import { eslintConfig } from '@aivillage/config'
```

## Migration Timeline

1. **Week 1**: Setup monorepo structure and tooling
2. **Week 2**: Create UI kit with shared components
3. **Week 3**: Migrate web app to Next.js + Expo Router
4. **Week 4**: Migrate mobile app to Expo
5. **Week 5**: Testing, optimization, and documentation
