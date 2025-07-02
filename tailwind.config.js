/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
    "*.{js,ts,jsx,tsx,mdx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        "logo-purple": "hsl(var(--logo-purple, 270 70% 40%))",
        "logo-cyan": "hsl(var(--logo-cyan, 190 70% 50%))",
        "logo-purple-dark": "hsl(var(--logo-purple-dark, 270 70% 30%))",
        "logo-cyan-dark": "hsl(var(--logo-cyan-dark, 190 70% 30%))",
        "space-gray": "hsl(var(--space-gray, 220 10% 10%))",
        "midnight-blue": "hsl(var(--midnight-blue, 230 20% 10%))",
        "nebula-purple": "hsl(var(--nebula-purple, 270 60% 35%))",
        "star-yellow": "hsl(var(--star-yellow, 30 70% 60%))",
        "cosmic-dark": "#1e1e2f", // Fallback for dark theme
        "cosmic-light": "#d1e8ff", // Fallback for light theme
      },
      backgroundImage: {
        "logo-gradient": "linear-gradient(135deg, hsl(var(--nebula-purple)), hsl(var(--star-yellow)))",
        "cosmic-gradient": "linear-gradient(180deg, hsl(var(--space-gray)), hsl(var(--midnight-blue)))",
        "logo-purple-radial": "radial-gradient(circle at center, hsl(var(--nebula-purple) / 0.2), transparent)",
        "logo-cyan-radial": "radial-gradient(circle at center, hsl(var(--logo-cyan) / 0.2), transparent)",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        "logo-sm": "0 1px 3px hsl(var(--nebula-purple) / 0.1), 0 1px 2px hsl(var(--star-yellow) / 0.05)",
        "logo-md": "0 4px 6px -1px hsl(var(--nebula-purple) / 0.1), 0 2px 4px -1px hsl(var(--star-yellow) / 0.05)",
        "logo-lg": "0 10px 15px -3px hsl(var(--nebula-purple) / 0.1), 0 4px 6px -2px hsl(var(--star-yellow) / 0.05)",
        "logo-xl": "0 20px 25px -5px hsl(var(--nebula-purple) / 0.15), 0 10px 10px -5px hsl(var(--star-yellow) / 0.1)",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
      fontSize: {
        xs: ["0.75rem", { lineHeight: "1rem" }],
        sm: ["0.875rem", { lineHeight: "1.25rem" }],
        base: ["1rem", { lineHeight: "1.5rem" }],
        lg: ["1.125rem", { lineHeight: "1.75rem" }],
        xl: ["1.25rem", { lineHeight: "1.75rem" }],
        "2xl": ["1.5rem", { lineHeight: "2rem" }],
        "3xl": ["1.875rem", { lineHeight: "2.25rem" }],
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        "pulse-slow": {
          "0%, 100%": { transform: "scale(1)", opacity: "1" },
          "50%": { transform: "scale(1.03)", opacity: "0.9" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        orbit: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        float: "float 6s ease-in-out infinite",
        "pulse-slow": "pulse-slow 4s ease-in-out infinite", // Fixed typo
        shimmer: "shimmer 1.5s ease-in-out infinite", // Adjusted for performance
        orbit: "orbit 8s linear infinite", // Adjusted for smoother orbit
      },
      transitionTimingFunction: {
        logo: "cubic-bezier(0.4, 0, 0.2, 1)",
      },
      transitionDelay: {
        300: "300ms",
        500: "500ms",
      },
      width: {
        "sidebar-open": "16rem", // Matches sidebar-open in globals.css
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    // Optional: Add tailwind-scrollbar for custom scrollbar support
    // require('tailwind-scrollbar'),
  ],
}