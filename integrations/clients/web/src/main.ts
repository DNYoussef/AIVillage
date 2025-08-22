import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createI18n } from 'vue-i18n'
import { registerSW } from 'virtual:pwa-register'

import App from './App.vue'
import router from './router'
import { useAppStore } from './stores/app'
import './style.css'

// Import translations
import en from './locales/en.json'
import es from './locales/es.json'
import fr from './locales/fr.json'
import de from './locales/de.json'
import ja from './locales/ja.json'
import zh from './locales/zh.json'

// Register service worker for PWA functionality
const updateSW = registerSW({
  onNeedRefresh() {
    // Show update notification to user
    const appStore = useAppStore()
    appStore.setUpdateAvailable(true)
  },
  onOfflineReady() {
    console.log('App ready to work offline')
  }
})

// Create i18n instance
const i18n = createI18n({
  legacy: false,
  locale: navigator.language?.split('-')[0] || 'en',
  fallbackLocale: 'en',
  messages: {
    en,
    es,
    fr,
    de,
    ja,
    zh
  }
})

// Create app instance
const app = createApp(App)

// Use plugins
app.use(createPinia())
app.use(router)
app.use(i18n)

// Global properties for accessibility
app.config.globalProperties.$announceToScreenReader = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
  const announcement = document.createElement('div')
  announcement.setAttribute('aria-live', priority)
  announcement.setAttribute('aria-atomic', 'true')
  announcement.className = 'sr-only'
  announcement.textContent = message

  document.body.appendChild(announcement)

  setTimeout(() => {
    document.body.removeChild(announcement)
  }, 1000)
}

// Error handler
app.config.errorHandler = (error, instance, info) => {
  console.error('Vue error:', error)
  console.error('Component instance:', instance)
  console.error('Error info:', info)

  // Announce error to screen readers
  if (instance?.appContext?.config?.globalProperties?.$announceToScreenReader) {
    instance.appContext.config.globalProperties.$announceToScreenReader(
      'An error occurred. Please check the console for details.',
      'assertive'
    )
  }
}

// Mount the app
app.mount('#app')

// Export updateSW for manual updates
export { updateSW }
