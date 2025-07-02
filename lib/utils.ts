import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  const merged = twMerge(clsx(inputs))

  // Optional: Debug invalid classes in development
  if (process.env.NODE_ENV === "development") {
    const invalidClasses = inputs
      .flat()
      .filter((cls) => typeof cls === "string" && !cls.trim())
      .join(", ")
    if (invalidClasses) {
      console.warn("Invalid or empty classes detected:", invalidClasses)
    }
  }

  return merged
}

// Memoized version for performance in frequent re-renders
import { useMemo } from "react"
export const useCn = (...inputs: ClassValue[]) => {
  return useMemo(() => twMerge(clsx(inputs)), [inputs])
}
