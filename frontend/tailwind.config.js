/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef9f6',
          100: '#d8f1ea',
          200: '#b5e2d7',
          300: '#89cdbd',
          400: '#5cb2a0',
          500: '#3e9987',
          600: '#2e7d6f',
          700: '#28645a',
          800: '#23514a',
          900: '#1f443e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 4px 24px -6px rgba(0, 0, 0, 0.12)',
        'glow': '0 0 30px -8px rgba(46, 125, 111, 0.45)',
        'glow-red': '0 0 34px -6px rgba(220, 38, 38, 0.55)',
      },
      keyframes: {
        'fade-in-up': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(16px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'pulse-ring': {
          '0%': { boxShadow: '0 0 0 0 rgba(220, 38, 38, 0.55)' },
          '100%': { boxShadow: '0 0 0 16px rgba(220, 38, 38, 0)' },
        },
        'typing-dot': {
          '0%, 60%, 100%': { transform: 'translateY(0)', opacity: '0.4' },
          '30%': { transform: 'translateY(-4px)', opacity: '1' },
        },
      },
      animation: {
        'fade-in-up': 'fade-in-up 0.35s ease-out both',
        'slide-in-right': 'slide-in-right 0.3s ease-out both',
        'pulse-ring': 'pulse-ring 1.4s ease-out infinite',
        'typing-dot': 'typing-dot 1.2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}