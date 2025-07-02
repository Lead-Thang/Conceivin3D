"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  type?: "text" | "number" | "color" | "range" | "file"
}

interface InputWithHelperProps extends InputProps {
  label?: string
  helperText?: string
  id?: string
}

const InputWithHelper = React.forwardRef<HTMLInputElement, InputWithHelperProps>(
  ({ label, helperText, id, className, type = "text", ...props }, ref) => {
    const inputId = id || props.name || `input-${Math.random().toString(36).substr(2, 9)}`
    
    return (
      <div className="space-y-1 w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-gray-300"
          >
            {label}
          </label>
        )}
        <input
          id={inputId}
          type={type}
          className={cn(
            "flex h-10 w-full rounded-md border border-slate-700 bg-gray-900/70 px-3 py-2 text-sm text-white placeholder:text-gray-500 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-logo-cyan focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200",
            type === "number" && "text-right",
            type === "color" && "w-10 h-10 p-0 rounded-full border-2 border-white/20",
            className
          )}
          ref={ref}
          {...props}
        />
        {helperText && (
          <p className="text-xs text-gray-400">{helperText}</p>
        )}
      </div>
    )
  }
)

InputWithHelper.displayName = "InputWithHelper"

export { InputWithHelper }
