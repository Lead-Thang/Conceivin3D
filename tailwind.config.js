/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--muted-foreground) / 0.3)',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        nebulaPurple: 'hsl(var(--nebula-purple))',
      },
      ringColor: {
        DEFAULT: 'hsl(var(--nebula-purple))',
        40: 'hsl(var(--nebula-purple) / 0.4)',
      },
      ringOffsetColor: {
        DEFAULT: 'hsl(var(--background))',
      },
      ringOffsetWidth: {
        DEFAULT: '2px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/postcss7-compat')({ stage: 2 }),
    require('tailwindcss/plugins/typography'),
    require('tailwindcss/plugins/forms'),
    require('tailwindcss/plugins/aspect-ratio'),
  ],
}