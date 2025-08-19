<template>
  <div class="max-w-4xl mx-auto">
    <!-- Page header with accessibility announcements -->
    <div class="mb-6">
      <h1 class="text-2xl font-bold text-gray-900" :id="chatTitleId">
        {{ $t('chat.title') }}
      </h1>
      <p class="mt-1 text-sm text-gray-600">
        {{ $t('chat.description') }}
      </p>
    </div>

    <!-- Chat configuration -->
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Agent preference -->
        <div>
          <label :for="agentSelectId" class="block text-sm font-medium text-gray-700 mb-1">
            {{ $t('chat.agentPreference') }}
          </label>
          <select
            :id="agentSelectId"
            v-model="chatConfig.agentPreference"
            class="block w-full rounded-md border-gray-300 text-base focus:border-primary-500 focus:ring-primary-500"
            :aria-describedby="`${agentSelectId}-desc`"
          >
            <option value="any">{{ $t('chat.agents.any') }}</option>
            <option value="king">{{ $t('chat.agents.king') }}</option>
            <option value="magi">{{ $t('chat.agents.magi') }}</option>
            <option value="sage">{{ $t('chat.agents.sage') }}</option>
            <option value="oracle">{{ $t('chat.agents.oracle') }}</option>
            <option value="navigator">{{ $t('chat.agents.navigator') }}</option>
          </select>
          <p :id="`${agentSelectId}-desc`" class="mt-1 text-xs text-gray-500">
            Choose which type of AI agent should respond to your messages
          </p>
        </div>

        <!-- Response mode -->
        <div>
          <label :for="modeSelectId" class="block text-sm font-medium text-gray-700 mb-1">
            {{ $t('chat.mode') }}
          </label>
          <select
            :id="modeSelectId"
            v-model="chatConfig.mode"
            class="block w-full rounded-md border-gray-300 text-base focus:border-primary-500 focus:ring-primary-500"
            :aria-describedby="`${modeSelectId}-desc`"
          >
            <option value="fast">{{ $t('chat.modes.fast') }}</option>
            <option value="balanced">{{ $t('chat.modes.balanced') }}</option>
            <option value="comprehensive">{{ $t('chat.modes.comprehensive') }}</option>
            <option value="creative">{{ $t('chat.modes.creative') }}</option>
          </select>
          <p :id="`${modeSelectId}-desc`" class="mt-1 text-xs text-gray-500">
            Choose how detailed and thorough the AI response should be
          </p>
        </div>
      </div>

      <!-- Action buttons -->
      <div class="flex justify-end mt-4 space-x-2">
        <button
          @click="clearConversation"
          :disabled="messages.length === 0"
          class="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ $t('chat.clear') }}
        </button>
        <button
          @click="newConversation"
          class="px-3 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
        >
          {{ $t('chat.newConversation') }}
        </button>
      </div>
    </div>

    <!-- Chat messages -->
    <div
      ref="messagesContainer"
      class="bg-white rounded-lg shadow-sm border border-gray-200 mb-4 max-h-96 overflow-y-auto"
      role="log"
      :aria-label="$t('chat.messagesLog')"
      :aria-live="isTyping ? 'polite' : 'off'"
    >
      <div class="p-4">
        <div v-if="messages.length === 0" class="text-center text-gray-500 py-8">
          <ChatIcon class="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>{{ $t('chat.noMessages') }}</p>
        </div>

        <div v-else class="space-y-4">
          <div
            v-for="(message, index) in messages"
            :key="message.id"
            class="flex"
            :class="message.role === 'user' ? 'justify-end' : 'justify-start'"
          >
            <div
              class="max-w-xs lg:max-w-md px-4 py-2 rounded-lg"
              :class="message.role === 'user'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-900'"
            >
              <div class="text-sm">
                {{ message.content }}
              </div>
              <div
                class="text-xs mt-1 opacity-75"
                :class="message.role === 'user' ? 'text-primary-100' : 'text-gray-500'"
              >
                <span v-if="message.role === 'assistant' && message.agent">
                  {{ message.agent }} •
                </span>
                <time :datetime="message.timestamp.toISOString()">
                  {{ formatTime(message.timestamp) }}
                </time>
                <span v-if="message.fromCache" class="ml-1" :title="$t('chat.fromCache')">
                  ⚡
                </span>
              </div>
            </div>
          </div>

          <!-- Typing indicator -->
          <div v-if="isTyping" class="flex justify-start">
            <div class="bg-gray-100 rounded-lg px-4 py-2 max-w-xs">
              <div class="flex space-x-1">
                <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
              </div>
              <div class="sr-only">{{ $t('chat.status.typing') }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Message input -->
    <form @submit.prevent="sendMessage" class="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div class="flex space-x-4">
        <div class="flex-1">
          <label :for="messageInputId" class="sr-only">
            {{ $t('chat.placeholder') }}
          </label>
          <textarea
            :id="messageInputId"
            ref="messageInput"
            v-model="currentMessage"
            :placeholder="$t('chat.placeholder')"
            :disabled="isTyping || !appStore.isOnline"
            :aria-describedby="messageInputId + '-status'"
            class="block w-full rounded-md border-gray-300 focus:border-primary-500 focus:ring-primary-500 disabled:bg-gray-50 disabled:text-gray-500"
            rows="3"
            @keydown.enter.exact="onEnterKey"
          />
          <div :id="messageInputId + '-status'" class="sr-only">
            <span v-if="isTyping">{{ $t('chat.status.typing') }}</span>
            <span v-else-if="!appStore.isOnline">{{ $t('chat.status.offline') }}</span>
            <span v-else-if="apiStore.isRateLimited">{{ $t('chat.status.rateLimit', { seconds: rateLimitSeconds }) }}</span>
          </div>
        </div>
        <div class="flex flex-col justify-end">
          <button
            type="submit"
            :disabled="!canSendMessage"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <SendIcon class="w-4 h-4" />
            <span class="sr-only">{{ $t('chat.send') }}</span>
          </button>
        </div>
      </div>

      <!-- Status messages -->
      <div v-if="statusMessage" class="mt-2 text-sm" :class="statusMessage.type === 'error' ? 'text-red-600' : 'text-gray-600'">
        <div role="alert" :aria-live="statusMessage.type === 'error' ? 'assertive' : 'polite'">
          {{ statusMessage.text }}
        </div>
      </div>
    </form>

    <!-- Performance stats (for debugging) -->
    <div v-if="showStats" class="mt-4 p-4 bg-gray-50 rounded-lg text-sm">
      <h3 class="font-medium mb-2">Performance Statistics</h3>
      <dl class="grid grid-cols-2 gap-4">
        <div>
          <dt class="font-medium">Cache Hit Rate:</dt>
          <dd>{{ appStore.cacheHitRate }}%</dd>
        </div>
        <div>
          <dt class="font-medium">Active Requests:</dt>
          <dd>{{ apiStore.activeRequests.size }}</dd>
        </div>
        <div>
          <dt class="font-medium">Recent Success Rate:</dt>
          <dd>{{ apiStore.requestStats.successful }}/{{ apiStore.requestStats.total }}</dd>
        </div>
        <div>
          <dt class="font-medium">Average Response Time:</dt>
          <dd>{{ apiStore.requestStats.averageDuration }}ms</dd>
        </div>
      </dl>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, nextTick, watch, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppStore } from '@/stores/app'
import { useApiStore } from '@/stores/api'

// Simple icons (in a real app, you'd use a proper icon library)
const ChatIcon = {
  template: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.959 8.959 0 01-4.906-1.471L3 21l2.471-5.094A8.959 8.959 0 013 12c0-4.418 3.582-8 8-8s8 3.582 8 8z" /></svg>`
}

const SendIcon = {
  template: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>`
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  agent?: string
  fromCache?: boolean
}

interface StatusMessage {
  type: 'info' | 'error'
  text: string
}

const { t } = useI18n()
const appStore = useAppStore()
const apiStore = useApiStore()

// Reactive data
const messages = ref<ChatMessage[]>([])
const currentMessage = ref('')
const isTyping = ref(false)
const conversationId = ref<string | undefined>()
const statusMessage = ref<StatusMessage | null>(null)
const showStats = ref(process.env.NODE_ENV === 'development')

// Chat configuration
const chatConfig = reactive({
  agentPreference: 'any' as const,
  mode: 'balanced' as const
})

// Component refs
const messageInput = ref<HTMLTextAreaElement>()
const messagesContainer = ref<HTMLElement>()

// Unique IDs for accessibility
const chatTitleId = `chat-title-${Math.random().toString(36).substr(2, 9)}`
const agentSelectId = `agent-select-${Math.random().toString(36).substr(2, 9)}`
const modeSelectId = `mode-select-${Math.random().toString(36).substr(2, 9)}`
const messageInputId = `message-input-${Math.random().toString(36).substr(2, 9)}`

// Computed properties
const canSendMessage = computed(() => {
  return currentMessage.value.trim().length > 0 &&
         !isTyping.value &&
         appStore.isOnline &&
         !apiStore.isRateLimited
})

const rateLimitSeconds = computed(() => {
  if (!apiStore.rateLimitInfo) return 0
  return Math.ceil((apiStore.rateLimitInfo.resetTime * 1000 - Date.now()) / 1000)
})

// Methods
const sendMessage = async () => {
  if (!canSendMessage.value) return

  const messageText = currentMessage.value.trim()
  const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

  // Add user message
  const userMessage: ChatMessage = {
    id: messageId,
    role: 'user',
    content: messageText,
    timestamp: new Date()
  }
  messages.value.push(userMessage)

  // Clear input and show typing
  currentMessage.value = ''
  isTyping.value = true
  statusMessage.value = null

  // Announce to screen readers
  announceToScreenReader(`Message sent: ${messageText}`)

  // Scroll to bottom
  scrollToBottom()

  try {
    // Send to API with caching disabled for real-time chat
    const response = await apiStore.sendChatMessage(messageText, {
      conversationId: conversationId.value,
      agentPreference: chatConfig.agentPreference === 'any' ? undefined : chatConfig.agentPreference,
      mode: chatConfig.mode,
      userContext: {
        device_type: 'desktop', // Could be detected
        network_type: appStore.isOnline ? 'wifi' : 'offline'
      }
    })

    // Update conversation ID
    if (response.conversation_id) {
      conversationId.value = response.conversation_id
    }

    // Add assistant response
    const assistantMessage: ChatMessage = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      role: 'assistant',
      content: response.response,
      timestamp: new Date(),
      agent: response.agent_used
    }
    messages.value.push(assistantMessage)

    // Announce response to screen readers
    announceToScreenReader(`Response received from ${response.agent_used}: ${response.response}`)

    // Show success status
    statusMessage.value = {
      type: 'info',
      text: `Response from ${response.agent_used} (${response.processing_time_ms}ms)`
    }

  } catch (error) {
    console.error('Chat error:', error)

    let errorMessage = t('error.generic')
    if (error instanceof Error) {
      errorMessage = error.message
    }

    statusMessage.value = {
      type: 'error',
      text: errorMessage
    }

    announceToScreenReader(`Error: ${errorMessage}`, 'assertive')
  } finally {
    isTyping.value = false
    scrollToBottom()

    // Focus back to input for keyboard users
    nextTick(() => {
      messageInput.value?.focus()
    })
  }
}

const clearConversation = () => {
  messages.value = []
  conversationId.value = undefined
  statusMessage.value = null
  announceToScreenReader(t('chat.conversationCleared'))
}

const newConversation = () => {
  clearConversation()
  // Focus input for immediate typing
  nextTick(() => {
    messageInput.value?.focus()
  })
}

const onEnterKey = (event: KeyboardEvent) => {
  if (!event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const formatTime = (timestamp: Date): string => {
  return new Intl.DateTimeFormat('default', {
    hour: '2-digit',
    minute: '2-digit'
  }).format(timestamp)
}

const announceToScreenReader = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
  const announcement = document.createElement('div')
  announcement.setAttribute('aria-live', priority)
  announcement.setAttribute('aria-atomic', 'true')
  announcement.className = 'sr-only'
  announcement.textContent = message

  document.body.appendChild(announcement)

  setTimeout(() => {
    if (document.body.contains(announcement)) {
      document.body.removeChild(announcement)
    }
  }, 1000)
}

// Auto-clear status messages
let statusTimeout: ReturnType<typeof setTimeout> | null = null
watch(statusMessage, (newStatus) => {
  if (statusTimeout) {
    clearTimeout(statusTimeout)
  }

  if (newStatus) {
    statusTimeout = setTimeout(() => {
      statusMessage.value = null
    }, 5000)
  }
})

// Initialize API client
onMounted(() => {
  // Initialize with any stored API key or use default configuration
  apiStore.initializeClient()

  // Focus message input
  messageInput.value?.focus()
})
</script>

<style scoped>
/* Screen reader only styles */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Smooth scrolling for reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  .overflow-y-auto {
    scroll-behavior: auto;
  }
}

/* High contrast support */
@media (prefers-contrast: high) {
  .bg-primary-600 {
    background-color: #1e3a8a;
  }

  .bg-gray-100 {
    background-color: #f9fafb;
    border: 1px solid #d1d5db;
  }
}

/* Focus visible for keyboard navigation */
button:focus-visible,
select:focus-visible,
textarea:focus-visible {
  outline: 2px solid #2563eb;
  outline-offset: 2px;
}
</style>
