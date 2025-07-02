// components/ui/model-viewer-skeleton.tsx
"use client"

import { Skeleton } from "./skeleton"

export function ModelViewerSkeleton() {
  return (
    <div className="relative h-[500px] w-full bg-black/50 rounded-xl flex items-center justify-center overflow-hidden">
      {/* Background Grid */}
      <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:60px_60px]" />

      {/* Spinner + Message */}
      <div className="flex flex-col items-center space-y-4 p-6">
        <div className="w-12 h-12 border-4 border-gray-600 border-t-logo-cyan rounded-full animate-spin" />
        <Skeleton className="h-4 w-32 rounded" />
      </div>
    </div>
  )
}
