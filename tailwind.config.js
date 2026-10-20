/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        canvas: {
          dark: '#090A0F',
          light: '#F8F9FA'
        },
        card: {
          dark: '#12141C',
          light: '#FFFFFF'
        },
        border: {
          dark: 'rgba(255, 255, 255, 0.07)',
          light: 'rgba(0, 0, 0, 0.09)'
        },
        accent: {
          blue: '#3B82F6',
          indigo: '#6366F1',
          emerald: '#10B981',
          rose: '#F43F5E',
          amber: '#F59E0B',
          purple: '#A855F7',
          cyan: '#06B6D4'
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace']
      }
    },
  },
  plugins: [],
}
