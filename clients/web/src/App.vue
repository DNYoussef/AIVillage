<template>
  <div id="app" class="min-h-screen bg-gray-50">
    <!-- Skip to main content link for screen readers -->
    <a
      href="#main"
      class="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-50 bg-primary-600 text-white px-4 py-2 rounded-md"
    >
      {{ $t('accessibility.skipToMain') }}
    </a>

    <!-- Header -->
    <header class="bg-white shadow-sm border-b border-gray-200" role="banner">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <!-- Logo and title -->
          <div class="flex items-center space-x-4">
            <div class="flex-shrink-0">
              <h1 class="text-xl font-bold text-gray-900">
                {{ $t('app.title') }}
              </h1>
            </div>
          </div>

          <!-- Navigation and controls -->
          <div class="flex items-center space-x-4">
            <!-- Network status indicator -->
            <NetworkStatusIndicator />

            <!-- Language selector -->
            <LanguageSelector />

            <!-- Theme toggle -->
            <ThemeToggle />

            <!-- Accessibility menu -->
            <AccessibilityMenu />
          </div>
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main id="main" role="main" class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Loading indicator -->
      <LoadingIndicator v-if="appStore.isLoading" />

      <!-- Route view -->
      <RouterView v-else />

      <!-- Update notification -->
      <UpdateNotification v-if="appStore.updateAvailable" />
    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-gray-200 mt-auto" role="contentinfo">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div class="flex justify-between items-center">
          <p class="text-sm text-gray-500">
            {{ $t('app.footer.copyright') }}
          </p>
          <div class="flex space-x-6">
            <a
              href="https://github.com/DNYoussef/AIVillage"
              class="text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              :aria-label="$t('app.footer.github')"
            >
              <span class="sr-only">{{ $t('app.footer.github') }}</span>
              <!-- GitHub icon would go here -->
            </a>
          </div>
        </div>
      </div>
    </footer>

    <!-- Screen reader announcements -->
    <div id="announcements" aria-live="polite" aria-atomic="true" class="sr-only" />
  </div>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'
import { useAppStore } from '@/stores/app'
import NetworkStatusIndicator from '@/components/NetworkStatusIndicator.vue'
import LanguageSelector from '@/components/LanguageSelector.vue'
import ThemeToggle from '@/components/ThemeToggle.vue'
import AccessibilityMenu from '@/components/AccessibilityMenu.vue'
import LoadingIndicator from '@/components/LoadingIndicator.vue'
import UpdateNotification from '@/components/UpdateNotification.vue'

const appStore = useAppStore()
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

.focus\:not-sr-only:focus {
  position: static;
  width: auto;
  height: auto;
  padding: 0.5rem 1rem;
  margin: 0;
  overflow: visible;
  clip: auto;
  white-space: normal;
}

/* High contrast focus styles */
*:focus {
  outline: 2px solid #2563eb;
  outline-offset: 2px;
}

/* Ensure sufficient color contrast */
@media (prefers-contrast: high) {
  .text-gray-500 {
    color: #1f2937;
  }

  .text-gray-400 {
    color: #374151;
  }
}

/* Respect reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
