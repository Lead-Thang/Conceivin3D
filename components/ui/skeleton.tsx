"use client"

import { cn } from "@/lib/utils"
import type { HTMLAttributes } from "react"

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "shimmer" | "pulse"
}

export function Skeleton({ className, variant = "shimmer", ...props }: SkeletonProps) {
  const variants = {
    default: "animate-pulse bg-gray-700",
    shimmer:
      "animate-pulse bg-gradient-to-r from-gray-700 via-gray-600 to-gray-700 bg-[length:400%_100%] animate-shimmer",
    pulse: "animate-pulse bg-gray-600",
  }

  return (
    <div
      className={cn("rounded-md opacity-60", variants[variant], className)}
      role="status"
      aria-label="Loading content"
      {...props}
    />
  )
}
