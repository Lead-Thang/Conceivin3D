"use client"

import { cn } from "@/lib/utils"

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg"
  className?: string
  text?: string
}

export function LoadingSpinner({ size = "md", className, text }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "w-4 h-4 border-2",
    md: "w-8 h-8 border-4",
    lg: "w-12 h-12 border-4",
  }

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center h-full w-full min-h-[60px] gap-3",
        className,
      )}
      role="status"
      aria-live="polite"
      aria-label={text ? `Loading ${text}` : "Loading, please wait"}
    >
      <div
        className={cn(
          "rounded-full border-t-nebula-purple border-r-star-yellow border-l-transparent border-b-transparent animate-spin animate-shimmer",
          sizeClasses[size],
        )}
        style={{
          borderImage: "linear-gradient(to right, hsl(var(--nebula-purple)), hsl(var(--star-yellow))) 1",
          borderWidth: sizeClasses[size].includes("border-2") ? "2px" : "4px",
        }}
      />
      {text && (
        <p
          className="text-sm text-muted-foreground animate-pulse"
          aria-hidden="true" // Text is decorative for screen readers
        >
          {text}
        </p>
      )}
    </div>
  )
}

// Keep the default export for backward compatibility
export default LoadingSpinner