"use client"

import { cn } from "@/lib/utils"

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-shimmer rounded-md bg-nebula-purple/20",
        className,
      )}
      role="status"
      aria-label="Loading content"
    />
  )
}

export function FeatureCardSkeleton() {
  return (
    <div className="feature-card p-6 rounded-xl bg-nebula-purple/50 backdrop-blur-sm shadow-lg">
      <Skeleton className="w-12 h-12 mb-4 rounded-lg bg-nebula-purple/30 animate-pulse" />
      <Skeleton className="h-6 w-3/4 mb-3 bg-nebula-purple/30" />
      <Skeleton className="h-4 w-full mb-2 bg-nebula-purple/30" />
      <Skeleton className="h-4 w-2/3 bg-nebula-purple/30" />
    </div>
  )
}

export function ModelViewerSkeleton() {
  return (
    <div
      className="h-full w-full bg-cosmic-gradient/70 rounded-xl flex items-center justify-center"
      role="status"
      aria-label="Initializing 3D viewer"
    >
      <div className="text-center">
        <div className="w-16 h-16 mx-auto mb-4 border-4 border-nebula-purple/30 border-t-star-yellow rounded-full animate-spin animate-orbit" />
        <p className="text-gray-400 text-sm animate-pulse">Initializing 3D Viewer...</p>
      </div>
    </div>
  )
}