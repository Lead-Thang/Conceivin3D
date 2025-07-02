"use client"

import * as React from "react"
import Image from "next/image"
import { Skeleton } from "./skeleton"
import { Card } from "./card"
import { cn } from "@/lib/utils"

interface CardWithImageLoaderProps {
  src: string
  alt: string
  isLoading?: boolean
  title?: string
  description?: string
  className?: string
  onClick?: () => void
}

export function CardWithImageLoader({
  src,
  alt,
  isLoading = false,
  title,
  description,
  className,
  onClick,
}: CardWithImageLoaderProps) {
  const [imageLoading, setImageLoading] = React.useState(true)
  const [imageError, setImageError] = React.useState(false)

  return (
    <Card
      className={cn(
        "relative overflow-hidden group cursor-pointer transition-all duration-300 hover:shadow-lg",
        className,
      )}
      onClick={onClick}
    >
      <div className="aspect-square w-full relative">
        {(isLoading || imageLoading) && !imageError ? (
          <Skeleton className="w-full h-full" />
        ) : imageError ? (
          <div className="w-full h-full bg-gray-800 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <div className="w-12 h-12 mx-auto mb-2 opacity-50">📷</div>
              <p className="text-xs">Image not available</p>
            </div>
          </div>
        ) : (
          <Image
            src={src || "/placeholder.svg"}
            alt={alt}
            fill
            className="object-cover transition-transform duration-500 group-hover:scale-105"
            onLoad={() => setImageLoading(false)}
            onError={() => {
              setImageLoading(false)
              setImageError(true)
            }}
          />
        )}
      </div>

      {(title || description) && (
        <div className="p-6 pt-4">
          {title && (
            <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-logo-cyan transition-colors">
              {title}
            </h3>
          )}
          {description && (
            <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">{description}</p>
          )}
        </div>
      )}
    </Card>
  )
}
