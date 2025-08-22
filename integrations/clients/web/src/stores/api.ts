import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { AIVillageClient, type Configuration, APIError, RateLimitError } from '@aivillage'
import { useAppStore } from './app'

export interface ApiRequestOptions {
  useCache?: boolean
  cacheTtl?: number
  idempotencyKey?: string
  retries?: number
}

export const useApiStore = defineStore('api', () => {
  const appStore = useAppStore()

  // API client configuration
  const config = ref<Configuration>({
    basePath: import.meta.env.VITE_API_BASE_URL || 'https://api.aivillage.io/v1',
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'AIVillage-Web-Client/1.0.0'
    }
  })

  // API client instance
  const client = ref<AIVillageClient | null>(null)

  // Request state
  const activeRequests = ref(new Set<string>())
  const requestHistory = ref<Array<{
    id: string
    method: string
    url: string
    timestamp: number
    duration?: number
    status?: number
    error?: string
  }>>([])

  // Rate limiting state
  const rateLimitInfo = ref<{
    limit: number
    remaining: number
    resetTime: number
  } | null>(null)

  // Initialize API client
  const initializeClient = (apiKey?: string, baseUrl?: string) => {
    const clientConfig: Configuration = {
      ...config.value,
      ...(baseUrl && { basePath: baseUrl }),
      ...(apiKey && { accessToken: apiKey })
    }

    client.value = new AIVillageClient(clientConfig)
    config.value = clientConfig
  }

  // Generic API request wrapper with caching and error handling
  const makeRequest = async <T>(
    requestFn: () => Promise<T>,
    cacheKey?: string,
    options: ApiRequestOptions = {}
  ): Promise<T> => {
    const {
      useCache = true,
      cacheTtl = 5 * 60 * 1000, // 5 minutes default
      idempotencyKey,
      retries = 3
    } = options

    const requestId = Math.random().toString(36).substr(2, 9)
    const startTime = Date.now()

    // Check cache first
    if (useCache && cacheKey) {
      const cached = appStore.getCache<T>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Track active request
    activeRequests.value.add(requestId)

    try {
      appStore.setLoading(true)

      let lastError: Error | null = null
      let attempt = 0

      while (attempt < retries) {
        try {
          const result = await requestFn()

          // Cache successful result
          if (useCache && cacheKey) {
            appStore.setCache(cacheKey, result, cacheTtl)
          }

          // Record successful request
          const duration = Date.now() - startTime
          requestHistory.value.push({
            id: requestId,
            method: 'API',
            url: cacheKey || 'unknown',
            timestamp: startTime,
            duration,
            status: 200
          })

          return result

        } catch (error) {
          lastError = error as Error
          attempt++

          if (error instanceof RateLimitError) {
            // Update rate limit info
            rateLimitInfo.value = {
              limit: parseInt(error.headers['x-ratelimit-limit'] || '0'),
              remaining: parseInt(error.headers['x-ratelimit-remaining'] || '0'),
              resetTime: parseInt(error.headers['x-ratelimit-reset'] || '0')
            }

            // Wait for retry-after period before retrying
            if (attempt < retries) {
              await new Promise(resolve => setTimeout(resolve, error.response.retry_after * 1000))
            }
          } else if (error instanceof APIError) {
            // Don't retry client errors (4xx)
            if (error.status >= 400 && error.status < 500) {
              break
            }

            // Exponential backoff for server errors
            if (attempt < retries) {
              const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000)
              await new Promise(resolve => setTimeout(resolve, backoffMs))
            }
          } else {
            // Network or other errors - exponential backoff
            if (attempt < retries) {
              const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000)
              await new Promise(resolve => setTimeout(resolve, backoffMs))
            }
          }
        }
      }

      // All retries exhausted
      if (lastError) {
        const duration = Date.now() - startTime
        const errorMessage = lastError instanceof APIError
          ? lastError.response.detail
          : lastError.message

        requestHistory.value.push({
          id: requestId,
          method: 'API',
          url: cacheKey || 'unknown',
          timestamp: startTime,
          duration,
          status: lastError instanceof APIError ? lastError.status : 0,
          error: errorMessage
        })

        appStore.addError(`API request failed: ${errorMessage}`)
        throw lastError
      }

      throw new Error('Request failed with no error details')

    } finally {
      activeRequests.value.delete(requestId)
      appStore.setLoading(activeRequests.value.size > 0)
    }
  }

  // Health check
  const checkHealth = async () => {
    if (!client.value) throw new Error('API client not initialized')

    return makeRequest(
      () => client.value!.health.getHealth(),
      'health-check',
      { cacheTtl: 30 * 1000 } // 30 seconds cache
    )
  }

  // Chat with agents
  const sendChatMessage = async (
    message: string,
    options: {
      conversationId?: string
      agentPreference?: 'king' | 'magi' | 'sage' | 'oracle' | 'navigator' | 'any'
      mode?: 'fast' | 'balanced' | 'comprehensive' | 'creative'
      userContext?: any
    } = {}
  ) => {
    if (!client.value) throw new Error('API client not initialized')

    const idempotencyKey = `chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

    return makeRequest(
      () => client.value!.chat.chat({
        message,
        conversation_id: options.conversationId,
        agent_preference: options.agentPreference || 'any',
        mode: options.mode || 'balanced',
        user_context: options.userContext
      }, { idempotencyKey }),
      `chat-${message.slice(0, 50)}-${JSON.stringify(options)}`,
      {
        useCache: false, // Chat responses should not be cached
        idempotencyKey
      }
    )
  }

  // RAG query
  const processQuery = async (
    query: string,
    options: {
      mode?: 'fast' | 'balanced' | 'comprehensive' | 'creative' | 'analytical'
      includeSources?: boolean
      maxResults?: number
      userId?: string
    } = {}
  ) => {
    if (!client.value) throw new Error('API client not initialized')

    const cacheKey = appStore.getCacheKey('rag-query', { query, ...options })

    return makeRequest(
      () => client.value!.rag.processQuery({
        query,
        mode: options.mode || 'balanced',
        include_sources: options.includeSources ?? true,
        max_results: options.maxResults || 10,
        user_id: options.userId
      }),
      cacheKey,
      {
        cacheTtl: 10 * 60 * 1000 // 10 minutes cache for RAG queries
      }
    )
  }

  // List agents
  const listAgents = async (options: {
    category?: 'governance' | 'infrastructure' | 'knowledge' | 'culture' | 'economy' | 'language' | 'health'
    availableOnly?: boolean
  } = {}) => {
    if (!client.value) throw new Error('API client not initialized')

    const cacheKey = appStore.getCacheKey('agents-list', options)

    return makeRequest(
      () => client.value!.agents.listAgents(options),
      cacheKey,
      {
        cacheTtl: 2 * 60 * 1000 // 2 minutes cache
      }
    )
  }

  // Assign agent task
  const assignAgentTask = async (
    agentId: string,
    taskDescription: string,
    options: {
      priority?: 'low' | 'medium' | 'high' | 'urgent'
      timeoutSeconds?: number
      context?: any
    } = {}
  ) => {
    if (!client.value) throw new Error('API client not initialized')

    const idempotencyKey = `task-${agentId}-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`

    return makeRequest(
      () => client.value!.agents.assignAgentTask(agentId, {
        task_description: taskDescription,
        priority: options.priority || 'medium',
        timeout_seconds: options.timeoutSeconds || 300,
        context: options.context
      }, { idempotencyKey }),
      undefined, // Don't cache task assignments
      {
        useCache: false,
        idempotencyKey
      }
    )
  }

  // P2P status
  const getP2PStatus = async () => {
    if (!client.value) throw new Error('API client not initialized')

    return makeRequest(
      () => client.value!.p2p.getP2PStatus(),
      'p2p-status',
      { cacheTtl: 30 * 1000 } // 30 seconds cache
    )
  }

  // List peers
  const listPeers = async (transport?: 'bitchat' | 'betanet' | 'all') => {
    if (!client.value) throw new Error('API client not initialized')

    return makeRequest(
      () => client.value!.p2p.listPeers(transport),
      `p2p-peers-${transport || 'all'}`,
      { cacheTtl: 30 * 1000 } // 30 seconds cache
    )
  }

  // Digital twin profile
  const getDigitalTwinProfile = async () => {
    if (!client.value) throw new Error('API client not initialized')

    return makeRequest(
      () => client.value!.digitalTwin.getDigitalTwinProfile(),
      'digital-twin-profile',
      { cacheTtl: 60 * 1000 } // 1 minute cache
    )
  }

  // Update digital twin data
  const updateDigitalTwinData = async (data: any) => {
    if (!client.value) throw new Error('API client not initialized')

    const idempotencyKey = `dt-update-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`

    return makeRequest(
      () => client.value!.digitalTwin.updateDigitalTwinData(data, { idempotencyKey }),
      undefined, // Don't cache updates
      {
        useCache: false,
        idempotencyKey
      }
    )
  }

  // Computed properties
  const isRateLimited = computed(() => {
    if (!rateLimitInfo.value) return false
    return rateLimitInfo.value.remaining <= 0 && Date.now() < rateLimitInfo.value.resetTime * 1000
  })

  const hasActiveRequests = computed(() => activeRequests.value.size > 0)

  const requestStats = computed(() => {
    const recent = requestHistory.value.filter(req => Date.now() - req.timestamp < 60 * 1000)
    const successful = recent.filter(req => !req.error)
    const failed = recent.filter(req => req.error)

    return {
      total: recent.length,
      successful: successful.length,
      failed: failed.length,
      averageDuration: successful.length > 0
        ? Math.round(successful.reduce((sum, req) => sum + (req.duration || 0), 0) / successful.length)
        : 0
    }
  })

  // Clear request history
  const clearRequestHistory = () => {
    requestHistory.value = []
  }

  // Reset rate limit info
  const resetRateLimitInfo = () => {
    rateLimitInfo.value = null
  }

  return {
    // State
    config,
    client,
    activeRequests,
    requestHistory,
    rateLimitInfo,

    // Actions
    initializeClient,
    makeRequest,
    checkHealth,
    sendChatMessage,
    processQuery,
    listAgents,
    assignAgentTask,
    getP2PStatus,
    listPeers,
    getDigitalTwinProfile,
    updateDigitalTwinData,
    clearRequestHistory,
    resetRateLimitInfo,

    // Computed
    isRateLimited,
    hasActiveRequests,
    requestStats
  }
})
