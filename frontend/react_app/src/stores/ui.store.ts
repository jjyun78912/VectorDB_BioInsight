/**
 * UI Store
 * Global state for UI preferences and modals
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

type ModalType = 'rnaseq-upload' | 'paper-agent' | 'settings' | null;

interface UIState {
  // Theme
  theme: 'light' | 'dark' | 'system';

  // Language
  language: 'ko' | 'en';

  // Modals
  activeModal: ModalType;
  modalData: Record<string, unknown> | null;

  // Sidebar
  sidebarOpen: boolean;

  // Notifications
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'info' | 'warning';
    message: string;
    timestamp: number;
  }>;

  // Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setLanguage: (language: 'ko' | 'en') => void;
  openModal: (modal: ModalType, data?: Record<string, unknown>) => void;
  closeModal: () => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  addNotification: (type: 'success' | 'error' | 'info' | 'warning', message: string) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

const generateId = () => `notif_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set, get) => ({
        theme: 'system',
        language: 'ko',
        activeModal: null,
        modalData: null,
        sidebarOpen: true,
        notifications: [],

        setTheme: (theme) => set({ theme }),

        setLanguage: (language) => set({ language }),

        openModal: (modal, data = null) =>
          set({
            activeModal: modal,
            modalData: data,
          }),

        closeModal: () =>
          set({
            activeModal: null,
            modalData: null,
          }),

        toggleSidebar: () =>
          set((state) => ({ sidebarOpen: !state.sidebarOpen })),

        setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),

        addNotification: (type, message) => {
          const notification = {
            id: generateId(),
            type,
            message,
            timestamp: Date.now(),
          };
          set((state) => ({
            notifications: [...state.notifications, notification],
          }));

          // Auto-remove after 5 seconds
          setTimeout(() => {
            get().removeNotification(notification.id);
          }, 5000);
        },

        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
          })),

        clearNotifications: () => set({ notifications: [] }),
      }),
      {
        name: 'bioinsight-ui',
        partialize: (state) => ({
          theme: state.theme,
          language: state.language,
          sidebarOpen: state.sidebarOpen,
        }),
      }
    ),
    { name: 'UIStore' }
  )
);
