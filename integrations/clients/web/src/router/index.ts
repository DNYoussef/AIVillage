import { createRouter, createWebHistory } from 'vue-router'
import { useAppStore } from '@/stores/app'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      redirect: '/chat'
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('@/views/ChatView.vue'),
      meta: {
        title: 'Chat with AI Agents',
        description: 'Interact with AIVillage\'s specialized AI agents'
      }
    },
    {
      path: '/query',
      name: 'query',
      component: () => import('@/views/QueryView.vue'),
      meta: {
        title: 'RAG Query System',
        description: 'Advanced knowledge retrieval and analysis'
      }
    },
    {
      path: '/agents',
      name: 'agents',
      component: () => import('@/views/AgentsView.vue'),
      meta: {
        title: 'AI Agent Management',
        description: 'Manage and monitor specialized AI agents'
      }
    },
    {
      path: '/p2p',
      name: 'p2p',
      component: () => import('@/views/P2PView.vue'),
      meta: {
        title: 'P2P Network Status',
        description: 'Monitor peer-to-peer network connections'
      }
    },
    {
      path: '/digital-twin',
      name: 'digital-twin',
      component: () => import('@/views/DigitalTwinView.vue'),
      meta: {
        title: 'Digital Twin Assistant',
        description: 'Personal AI assistant and privacy settings'
      }
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('@/views/SettingsView.vue'),
      meta: {
        title: 'Application Settings',
        description: 'Configure application preferences and accessibility'
      }
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'not-found',
      component: () => import('@/views/NotFoundView.vue'),
      meta: {
        title: 'Page Not Found',
        description: 'The requested page could not be found'
      }
    }
  ],
  // Accessibility improvements
  scrollBehavior(to, from, savedPosition) {
    // Respect user's reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (savedPosition) {
      return savedPosition
    } else if (to.hash) {
      return {
        el: to.hash,
        behavior: prefersReducedMotion ? 'auto' : 'smooth'
      }
    } else {
      return {
        top: 0,
        behavior: prefersReducedMotion ? 'auto' : 'smooth'
      }
    }
  }
})

// Global navigation guards
router.beforeEach((to, from, next) => {
  const appStore = useAppStore()

  // Update document title for screen readers
  if (to.meta?.title) {
    document.title = `${to.meta.title} - AIVillage`
  }

  // Update meta description
  if (to.meta?.description) {
    const metaDescription = document.querySelector('meta[name="description"]')
    if (metaDescription) {
      metaDescription.setAttribute('content', to.meta.description as string)
    }
  }

  // Announce route change to screen readers
  const routeName = (to.meta?.title as string) || to.name?.toString() || 'page'
  const announcement = `Navigated to ${routeName}`

  // Delay announcement slightly to avoid conflicts with page load
  setTimeout(() => {
    const announcer = document.createElement('div')
    announcer.setAttribute('aria-live', 'polite')
    announcer.setAttribute('aria-atomic', 'true')
    announcer.className = 'sr-only'
    announcer.textContent = announcement

    document.body.appendChild(announcer)

    setTimeout(() => {
      if (document.body.contains(announcer)) {
        document.body.removeChild(announcer)
      }
    }, 1000)
  }, 100)

  next()
})

router.afterEach((to, from) => {
  // Focus management for accessibility
  nextTick(() => {
    // Skip to main content link functionality
    const skipLink = document.querySelector('a[href="#main"]') as HTMLElement
    if (skipLink && document.activeElement === skipLink) {
      const mainContent = document.getElementById('main')
      if (mainContent) {
        mainContent.focus()
        mainContent.scrollIntoView()
      }
    }
  })
})

// Helper function for nextTick (would be imported in real app)
function nextTick(callback: () => void) {
  setTimeout(callback, 0)
}

export default router
