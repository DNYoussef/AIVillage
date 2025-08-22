<template>
  <div class="relative">
    <button
      ref="triggerRef"
      @click="toggleMenu"
      @keydown="onTriggerKeydown"
      :aria-expanded="isOpen"
      :aria-haspopup="true"
      aria-label="Accessibility settings"
      class="inline-flex items-center p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-md"
    >
      <AccessibilityIcon class="w-5 h-5" />
      <span class="sr-only">{{ $t('accessibility.settings') }}</span>
    </button>

    <Teleport to="body">
      <div
        v-if="isOpen"
        ref="overlayRef"
        class="fixed inset-0 z-50 bg-black bg-opacity-25"
        @click="closeMenu"
        @keydown.escape="closeMenu"
      />
    </Teleport>

    <Transition
      enter-active-class="transition ease-out duration-100"
      enter-from-class="transform opacity-0 scale-95"
      enter-to-class="transform opacity-100 scale-100"
      leave-active-class="transition ease-in duration-75"
      leave-from-class="transform opacity-100 scale-100"
      leave-to-class="transform opacity-0 scale-95"
    >
      <div
        v-if="isOpen"
        ref="menuRef"
        role="dialog"
        :aria-labelledby="dialogTitleId"
        :aria-describedby="dialogDescId"
        class="absolute right-0 z-50 w-80 mt-2 bg-white rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none"
        @keydown="onMenuKeydown"
      >
        <div class="p-4">
          <h3 :id="dialogTitleId" class="text-lg font-medium text-gray-900 mb-4">
            {{ $t('accessibility.settings') }}
          </h3>
          <p :id="dialogDescId" class="text-sm text-gray-600 mb-4">
            {{ $t('accessibility.description') }}
          </p>

          <div class="space-y-4">
            <!-- Reduced Motion -->
            <div class="flex items-start">
              <div class="flex h-5 items-center">
                <input
                  :id="reducedMotionId"
                  v-model="localSettings.reducedMotion"
                  @change="updateSetting('reducedMotion', $event.target.checked)"
                  type="checkbox"
                  class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </div>
              <div class="ml-3 text-sm">
                <label :for="reducedMotionId" class="font-medium text-gray-700">
                  {{ $t('accessibility.reducedMotion') }}
                </label>
                <p class="text-gray-500">
                  {{ $t('accessibility.reducedMotionDesc') }}
                </p>
              </div>
            </div>

            <!-- High Contrast -->
            <div class="flex items-start">
              <div class="flex h-5 items-center">
                <input
                  :id="highContrastId"
                  v-model="localSettings.highContrast"
                  @change="updateSetting('highContrast', $event.target.checked)"
                  type="checkbox"
                  class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </div>
              <div class="ml-3 text-sm">
                <label :for="highContrastId" class="font-medium text-gray-700">
                  {{ $t('accessibility.highContrast') }}
                </label>
                <p class="text-gray-500">
                  {{ $t('accessibility.highContrastDesc') }}
                </p>
              </div>
            </div>

            <!-- Large Text -->
            <div class="flex items-start">
              <div class="flex h-5 items-center">
                <input
                  :id="largeTextId"
                  v-model="localSettings.largeText"
                  @change="updateSetting('largeText', $event.target.checked)"
                  type="checkbox"
                  class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </div>
              <div class="ml-3 text-sm">
                <label :for="largeTextId" class="font-medium text-gray-700">
                  {{ $t('accessibility.largeText') }}
                </label>
                <p class="text-gray-500">
                  {{ $t('accessibility.largeTextDesc') }}
                </p>
              </div>
            </div>

            <!-- Screen Reader Mode -->
            <div class="flex items-start">
              <div class="flex h-5 items-center">
                <input
                  :id="screenReaderId"
                  v-model="localSettings.screenReader"
                  @change="updateSetting('screenReader', $event.target.checked)"
                  type="checkbox"
                  class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </div>
              <div class="ml-3 text-sm">
                <label :for="screenReaderId" class="font-medium text-gray-700">
                  {{ $t('accessibility.screenReader') }}
                </label>
                <p class="text-gray-500">
                  {{ $t('accessibility.screenReaderDesc') }}
                </p>
              </div>
            </div>

            <!-- Focus Indicator -->
            <div>
              <label :for="focusIndicatorId" class="block text-sm font-medium text-gray-700 mb-2">
                {{ $t('accessibility.focusIndicator') }}
              </label>
              <select
                :id="focusIndicatorId"
                v-model="localSettings.focusIndicator"
                @change="updateSetting('focusIndicator', $event.target.value)"
                class="block w-full rounded-md border-gray-300 py-2 pl-3 pr-10 text-base focus:border-primary-500 focus:outline-none focus:ring-primary-500"
              >
                <option value="default">{{ $t('accessibility.focusDefault') }}</option>
                <option value="high-contrast">{{ $t('accessibility.focusHighContrast') }}</option>
                <option value="large">{{ $t('accessibility.focusLarge') }}</option>
              </select>
            </div>
          </div>

          <div class="mt-6 flex justify-end space-x-3">
            <button
              @click="resetSettings"
              class="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {{ $t('accessibility.reset') }}
            </button>
            <button
              @click="closeMenu"
              class="px-3 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {{ $t('accessibility.done') }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, watch, nextTick, computed } from 'vue'
import { onKeyStroke, useFocusTrap } from '@vueuse/core'
import { useAppStore } from '@/stores/app'
import { useI18n } from 'vue-i18n'

// Icon component (simplified)
const AccessibilityIcon = {
  template: `
    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z" />
    </svg>
  `
}

const { t } = useI18n()
const appStore = useAppStore()

// Reactive state
const isOpen = ref(false)
const triggerRef = ref<HTMLElement>()
const menuRef = ref<HTMLElement>()
const overlayRef = ref<HTMLElement>()

// Local settings that sync with store
const localSettings = reactive({
  reducedMotion: appStore.accessibilitySettings.reducedMotion,
  highContrast: appStore.accessibilitySettings.highContrast,
  largeText: appStore.accessibilitySettings.largeText,
  screenReader: appStore.accessibilitySettings.screenReader,
  focusIndicator: appStore.accessibilitySettings.focusIndicator
})

// Unique IDs for form elements
const dialogTitleId = `a11y-title-${Math.random().toString(36).substr(2, 9)}`
const dialogDescId = `a11y-desc-${Math.random().toString(36).substr(2, 9)}`
const reducedMotionId = `reduced-motion-${Math.random().toString(36).substr(2, 9)}`
const highContrastId = `high-contrast-${Math.random().toString(36).substr(2, 9)}`
const largeTextId = `large-text-${Math.random().toString(36).substr(2, 9)}`
const screenReaderId = `screen-reader-${Math.random().toString(36).substr(2, 9)}`
const focusIndicatorId = `focus-indicator-${Math.random().toString(36).substr(2, 9)}`

// Focus trap for the dialog
const { activate: activateFocusTrap, deactivate: deactivateFocusTrap } = useFocusTrap(menuRef, {
  immediate: false,
  allowOutsideClick: true,
  returnFocusOnDeactivate: true,
  fallbackFocus: triggerRef
})

// Menu control
const toggleMenu = async () => {
  isOpen.value = !isOpen.value

  if (isOpen.value) {
    await nextTick()
    activateFocusTrap()
    // Announce to screen readers
    announceToScreenReader(t('accessibility.menuOpened'))
  } else {
    deactivateFocusTrap()
  }
}

const closeMenu = () => {
  isOpen.value = false
  deactivateFocusTrap()
  triggerRef.value?.focus()
}

// Keyboard navigation
const onTriggerKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    toggleMenu()
  }
}

const onMenuKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    event.preventDefault()
    closeMenu()
  }
}

// Settings management
const updateSetting = (key: keyof typeof localSettings, value: any) => {
  localSettings[key] = value
  appStore.updateAccessibilitySettings({ [key]: value })

  // Apply immediate effects
  applyAccessibilitySettings()

  // Announce change to screen readers
  announceToScreenReader(t(`accessibility.${key}Updated`, { value }))
}

const resetSettings = () => {
  const defaultSettings = {
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    highContrast: window.matchMedia('(prefers-contrast: high)').matches,
    largeText: false,
    screenReader: false,
    focusIndicator: 'default' as const
  }

  Object.assign(localSettings, defaultSettings)
  appStore.updateAccessibilitySettings(defaultSettings)
  applyAccessibilitySettings()

  announceToScreenReader(t('accessibility.settingsReset'))
}

// Apply accessibility settings to DOM
const applyAccessibilitySettings = () => {
  const html = document.documentElement

  // Reduced motion
  if (localSettings.reducedMotion) {
    html.classList.add('reduce-motion')
  } else {
    html.classList.remove('reduce-motion')
  }

  // High contrast
  if (localSettings.highContrast) {
    html.classList.add('high-contrast')
  } else {
    html.classList.remove('high-contrast')
  }

  // Large text
  if (localSettings.largeText) {
    html.classList.add('large-text')
  } else {
    html.classList.remove('large-text')
  }

  // Screen reader mode
  if (localSettings.screenReader) {
    html.classList.add('screen-reader')
  } else {
    html.classList.remove('screen-reader')
  }

  // Focus indicator
  html.classList.remove('focus-default', 'focus-high-contrast', 'focus-large')
  html.classList.add(`focus-${localSettings.focusIndicator}`)
}

// Screen reader announcements
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

// Watch for changes in store settings
watch(
  () => appStore.accessibilitySettings,
  (newSettings) => {
    Object.assign(localSettings, newSettings)
    applyAccessibilitySettings()
  },
  { deep: true }
)

// Apply settings on mount
applyAccessibilitySettings()

// Close menu when clicking outside or pressing escape
onKeyStroke('Escape', () => {
  if (isOpen.value) {
    closeMenu()
  }
})
</script>

<style scoped>
/* Accessibility-specific styles */
.reduce-motion * {
  animation-duration: 0.01ms !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0.01ms !important;
}

.high-contrast {
  filter: contrast(150%);
}

.large-text {
  font-size: 1.125rem;
}

.large-text h1,
.large-text h2,
.large-text h3 {
  font-size: 1.25em;
}

.focus-high-contrast *:focus {
  outline: 3px solid #000 !important;
  outline-offset: 2px !important;
}

.focus-large *:focus {
  outline: 4px solid #2563eb !important;
  outline-offset: 4px !important;
}

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
</style>
