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
      },
    },
  },
  plugins: [
    require('@tailwindcss/postcss7-compat')({ stage: 2 }),
  ],
}