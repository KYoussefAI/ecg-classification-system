/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        mono:    ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        cardiac: {
          50:  '#f0f7ff',
          100: '#e0efff',
          200: '#b9dcff',
          300: '#7cbeff',
          400: '#3698ff',
          500: '#0a78ff',
          600: '#0056f5',
          700: '#0041e1',
          800: '#0435b6',
          900: '#09308f',
          950: '#071d57',
        },
        danger: {
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
        },
        success: {
          400: '#4ade80',
          500: '#22c55e',
        },
        warn: {
          400: '#fbbf24',
          500: '#f59e0b',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ecg-line':   'ecgLine 2s ease-in-out infinite',
        'fade-in':    'fadeIn 0.4s ease-out',
        'slide-up':   'slideUp 0.4s ease-out',
      },
      keyframes: {
        ecgLine: {
          '0%, 100%': { strokeDashoffset: '1000' },
          '50%':      { strokeDashoffset: '0' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
