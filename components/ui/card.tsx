"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

// Types
interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  isLoading?: boolean
  variant?: "default" | "elevated" | "glass"
  action?: () => void
  actionLabel?: string
}

// Base Styles
const baseStyles = {
  default: "bg-gray-900/40 border-logo-purple/20",
  elevated: "bg-gray-800/70 border-logo-cyan/20 shadow-xl hover:shadow-logo-cyan/20",
  glass: "bg-gray-900/60 border border-white/10 backdrop-blur-md shadow-lg",
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, isLoading, variant = "default", action, actionLabel, ...props }, ref) => {
    const Comp = action ? "button" : "div"
    return (
      <Comp
        ref={ref}
        className={cn(
          "rounded-lg border text-card-foreground shadow-sm transition-all duration-300",
          baseStyles[variant],
          isLoading && "animate-pulse",
          action && "cursor-pointer focus:outline-none focus:ring-2 focus:ring-logo-cyan",
          className
        )}
        aria-label={actionLabel}
        onClick={action}
        {...(action && { tabIndex: 0 })}
        {...props}
      />
    )
  }
)
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-bold leading-none tracking-tight",
      "bg-clip-text text-transparent bg-gradient-to-r from-logo-purple to-logo-cyan",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("p-6 pt-0", className)}
    {...props}
  />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "flex items-center p-6 pt-0 border-t border-gray-800 mt-auto",
      "transition-all duration-200 hover:bg-gray-900/50",
      className
    )}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
}
