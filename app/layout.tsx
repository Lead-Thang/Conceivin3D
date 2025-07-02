import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/toaster"
import { ErrorBoundary } from "@/components/error-boundary"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: {
    default: "Conceivin3D",
    template: "%s | Conceivin3D - Cosmic 3D Design",
  },
  description: "Launch your ideas into the cosmos with Conceivin3D, a professional 3D design and visualization platform for tech companies powered by AI.",
  keywords: ["3D modeling", "CAD", "AI design", "space tech", "engineering", "product design", "cosmic design"],
  authors: [{ name: "Conceivin3D Team" }],
  viewport: "width=device-width, initial-scale=1, maximum-scale=1",
  robots: "index, follow",
  generator: "v0.dev",
  themeColor: "#1e1e2f", // Dark space-inspired theme color
  manifest: "/manifest.json", // Optional: Add a manifest for PWA support
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon-16x16.png",
    apple: "/apple-touch-icon.png",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full">
      <body className={cn(inter.className, "h-full bg-cosmic-gradient/95 text-white overflow-x-hidden")}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem enableColorScheme disableTransitionOnChange>
          <ErrorBoundary>{children}</ErrorBoundary>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}