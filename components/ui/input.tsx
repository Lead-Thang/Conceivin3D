"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  type?: "text" | "number" | "color" | "range" | "file"
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type || "text"}
        step={type === "number" ? "0.01" : undefined}
        className={cn(
          "flex h-10 w-full rounded-md border border-slate-700 bg-gray-900/70 px-3 py-2 text-sm text-white placeholder:text-gray-500 shadow-sm transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-logo-cyan focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          type === "number" && "text-right",
          type === "color" && "w-10 h-10 p-0 rounded-full border-2 border-white/20",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)

Input.displayName = "Input"

// Optional: Input with Label + Helper Text
interface InputWithHelperProps extends InputProps {
  label?: string
  helperText?: string
}

export const InputWithHelper = React.forwardRef<HTMLInputElement, InputWithHelperProps>(
  ({ label, helperText, ...props }, ref) => (
    <div className="space-y-1">
      {label && (
        <label className="text-sm font-medium text-gray-300 block">{label}</label>
      )}
      <Input {...props} ref={ref} />
      {helperText && (
        <p className="text-xs text-gray-500">{helperText}</p>
      )}
    </div>
  )
)

export { Input }
