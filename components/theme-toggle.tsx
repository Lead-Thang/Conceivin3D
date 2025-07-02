"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  // Smooth transition classes
  const sunClasses = cn(
    "h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all duration-300",
    theme === "dark" && "rotate-90 scale-0",
  )
  const moonClasses = cn(
    "absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all duration-300",
    theme === "dark" && "rotate-0 scale-100",
  )

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setTheme(theme === "light" ? "dark" : "light")}
      aria-label="Toggle theme"
      className="w-9 px-0 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-logo-purple/50"
    >
      <Sun className={sunClasses} />
      <Moon className={moonClasses} />
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}
