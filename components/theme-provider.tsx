"use client"

import { ThemeProvider as NextThemesProvider } from "next-themes"
import type { ThemeProviderProps, ThemeProvider as NextThemeProviderType } from "next-themes"
import type { FC, ReactNode } from "react"
import { useEffect } from "react"

interface CustomThemeProviderProps extends ThemeProviderProps {
  children: ReactNode
}

/**
 * ThemeProvider component that wraps next-themes with custom enhancements
 * Supports space-themed variants and ensures color scheme consistency.
 */
export const ThemeProvider: FC<CustomThemeProviderProps> = ({ children, ...props }) => {
  useEffect(() => {
    // Force color scheme update on mount to avoid hydration mismatches
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const handleThemeChange = () => {
      document.documentElement.style.colorScheme = media.matches ? "dark" : "light"
    }
    handleThemeChange()
    media.addEventListener("change", handleThemeChange)
    return () => media.removeEventListener("change", handleThemeChange)
  }, [])

  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="cosmic-dark" // Default to a space-inspired dark theme
      enableSystem
      disableTransitionOnChange
      themes={["cosmic-dark", "cosmic-light", "system"]} // Custom themes
      {...props}
    >
      {children}
    </NextThemesProvider>
  )
}