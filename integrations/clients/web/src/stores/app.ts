import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useOnline, usePreferredColorScheme } from '@vueuse/core'

export interface CacheEntry<T = any> {
  data: T
  timestamp: number
  ttl: number
  key: string
}

export interface AccessibilitySettings {
  reducedMotion: boolean
  highContrast: boolean
  largeText: boolean
  screenReader: boolean
  focusIndicator: 'default' | 'high-contrast' | 'large'
}

export const useAppStore = defineStore('app', () => {
  // Core app state
  const isLoading = ref(false)
  const updateAvailable = ref(false)
  const currentLanguage = ref('en')
  const online = useOnline()
  const preferredColorScheme = usePreferredColorScheme()

  // Theme state
  const theme = ref<'light' | 'dark' | 'auto'>('auto')
  const isDark = computed(() => {
    if (theme.value === 'auto') {
      return preferredColorScheme.value === 'dark'
    }
    return theme.value === 'dark'
  })

  // Accessibility state
  const accessibilitySettings = ref<AccessibilitySettings>({
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    highContrast: window.matchMedia('(prefers-contrast: high)').matches,
    largeText: false,
    screenReader: false,
    focusIndicator: 'default'
  })

  // Cache state
  const cache = ref(new Map<string, CacheEntry>())
  const cacheStats = ref({
    hits: 0,
    misses: 0,
    expired: 0,
    size: 0
  })

  // Error state
  const errors = ref<Array<{ id: string; message: string; timestamp: number }>>([])

  // Actions
  const setLoading = (loading: boolean) => {
    isLoading.value = loading
  }

  const setUpdateAvailable = (available: boolean) => {
    updateAvailable.value = available
  }

  const setLanguage = (lang: string) => {
    currentLanguage.value = lang
    localStorage.setItem('aivillage-language', lang)
  }

  const setTheme = (newTheme: 'light' | 'dark' | 'auto') => {
    theme.value = newTheme
    localStorage.setItem('aivillage-theme', newTheme)

    // Apply theme to document
    if (newTheme === 'dark' || (newTheme === 'auto' && preferredColorScheme.value === 'dark')) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  const updateAccessibilitySettings = (settings: Partial<AccessibilitySettings>) => {
    accessibilitySettings.value = { ...accessibilitySettings.value, ...settings }
    localStorage.setItem('aivillage-a11y', JSON.stringify(accessibilitySettings.value))
  }

  // Cache management
  const getCacheKey = (url: string, params?: Record<string, any>): string => {
    const base = url
    if (!params) return base

    const paramString = Object.keys(params)
      .sort()
      .map(key => `${key}=${JSON.stringify(params[key])}`)
      .join('&')

    return `${base}?${paramString}`
  }

  const setCache = <T>(key: string, data: T, ttlMs: number = 5 * 60 * 1000) => {
    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      ttl: ttlMs,
      key
    }

    cache.value.set(key, entry)
    cacheStats.value.size = cache.value.size
  }

  const getCache = <T>(key: string): T | null => {
    const entry = cache.value.get(key)
    if (!entry) {
      cacheStats.value.misses++
      return null
    }

    // Check expiration
    if (Date.now() - entry.timestamp > entry.ttl) {
      cache.value.delete(key)
      cacheStats.value.expired++
      cacheStats.value.size = cache.value.size
      return null
    }

    cacheStats.value.hits++
    return entry.data as T
  }

  const clearCache = (pattern?: RegExp) => {
    if (!pattern) {
      cache.value.clear()
    } else {
      for (const [key] of cache.value) {
        if (pattern.test(key)) {
          cache.value.delete(key)
        }
      }
    }
    cacheStats.value.size = cache.value.size
  }

  const cleanExpiredCache = () => {
    const now = Date.now()
    let cleaned = 0

    for (const [key, entry] of cache.value) {
      if (now - entry.timestamp > entry.ttl) {
        cache.value.delete(key)
        cleaned++
      }
    }

    cacheStats.value.expired += cleaned
    cacheStats.value.size = cache.value.size
  }

  // Error management
  const addError = (message: string) => {
    const id = Math.random().toString(36).substr(2, 9)
    errors.value.push({
      id,
      message,
      timestamp: Date.now()
    })

    // Auto-remove after 10 seconds
    setTimeout(() => {
      removeError(id)
    }, 10000)

    return id
  }

  const removeError = (id: string) => {
    const index = errors.value.findIndex(error => error.id === id)
    if (index > -1) {
      errors.value.splice(index, 1)
    }
  }

  const clearErrors = () => {
    errors.value = []
  }

  // Computed getters
  const cacheHitRate = computed(() => {
    const total = cacheStats.value.hits + cacheStats.value.misses
    return total > 0 ? (cacheStats.value.hits / total * 100).toFixed(1) : '0.0'
  })

  const isOnline = computed(() => online.value)

  // Initialize from localStorage
  const initialize = () => {
    // Language
    const savedLanguage = localStorage.getItem('aivillage-language')
    if (savedLanguage) {
      currentLanguage.value = savedLanguage
    }

    // Theme
    const savedTheme = localStorage.getItem('aivillage-theme') as 'light' | 'dark' | 'auto'
    if (savedTheme) {
      setTheme(savedTheme)
    }

    // Accessibility settings
    const savedA11y = localStorage.getItem('aivillage-a11y')
    if (savedA11y) {
      try {
        const parsed = JSON.parse(savedA11y)
        accessibilitySettings.value = { ...accessibilitySettings.value, ...parsed }
      } catch (e) {
        console.warn('Failed to parse accessibility settings from localStorage')
      }
    }

    // Clean expired cache entries every 5 minutes
    setInterval(cleanExpiredCache, 5 * 60 * 1000)
  }

  return {
    // State
    isLoading,
    updateAvailable,
    currentLanguage,
    theme,
    isDark,
    accessibilitySettings,
    cache,
    cacheStats,
    errors,

    // Actions
    setLoading,
    setUpdateAvailable,
    setLanguage,
    setTheme,
    updateAccessibilitySettings,
    getCacheKey,
    setCache,
    getCache,
    clearCache,
    cleanExpiredCache,
    addError,
    removeError,
    clearErrors,
    initialize,

    // Computed
    cacheHitRate,
    isOnline
  }
})
