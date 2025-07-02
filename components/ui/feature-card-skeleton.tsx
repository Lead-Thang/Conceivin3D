// components/ui/feature-card-skeleton.tsx
"use client"

import { Skeleton } from "./skeleton"

export function FeatureCardSkeleton() {
  return (
    <div className="p-6 rounded-xl bg-gray-900/30 backdrop-blur-sm border border-logo-purple/20 animate-pulse">
      <div className="flex flex-col space-y-4">
        <Skeleton className="w-12 h-12 rounded-lg" />
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
      </div>
    </div>
  )
}
