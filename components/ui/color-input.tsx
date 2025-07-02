"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { useCallback } from "react"

interface ColorInputProps {
  label?: string
  helperText?: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  id?: string
  className?: string
}

export function ColorInput({ label, helperText, value, onChange, id, className }: ColorInputProps) {
  const inputId = id || `color-input-${Math.random().toString(36).substr(2, 9)}`

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e)
  }, [onChange])

  return (
    <div className="space-y-1 w-full">
      {/* Label */}
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium text-gray-300">
          {label}
        </label>
      )}

      {/* Input + Swatch */}
      <div className="flex items-center space-x-2">
        <input
          id={inputId}
          type="color"
          value={value}
          onChange={handleChange}
          className={cn(
            "w-full h-8 rounded border border-slate-600 bg-slate-700 cursor-pointer",
            className
          )}
        />
        <span
          className="inline-block w-6 h-6 rounded border border-white/20"
          style={{ backgroundColor: value }}
          title={`Current color: ${value}`}
        ></span>
      </div>

      {/* Helper Text + Reset */}
      {(helperText || true) && (
        <div className="flex justify-between items-center">
          {helperText && <p className="text-xs text-gray-400">{helperText}</p>}
          <button
            type="button"
            onClick={() =>
              handleChange({
                target: { value: "#8b5cf6" },
              } as React.ChangeEvent<HTMLInputElement>)
            }
            className="text-xs text-gray-400 hover:text-logo-cyan"
          >
            Reset
          </button>
        </div>
      )}
    </div>
  )
}
