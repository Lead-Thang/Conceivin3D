"use client"

import type * as React from "react"
import { useToast } from "@/hooks/use-toast"
import { Toast, ToastTitle, ToastDescription, ToastClose } from "@/components/ui/toast"

interface ToastProviderProps {
  children: React.ReactNode
}

export function ToastProvider({ children }: ToastProviderProps) {
  const { toasts, dismiss } = useToast()

  return (
    <>
      {children}

      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2">
        {toasts.map(({ id, title, description, action, open, variant = "default" }) => (
          <Toast
            key={id}
            open={open}
            onOpenChange={(isOpen) => {
              if (!isOpen) {
                dismiss(id)
              }
            }}
            variant={variant}
          >
            <div>
              {title && <ToastTitle>{title}</ToastTitle>}
              {description && <ToastDescription>{description}</ToastDescription>}
              {action}
            </div>
            <ToastClose />
          </Toast>
        ))}
      </div>
    </>
  )
}
