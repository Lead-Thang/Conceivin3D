"use client"

import React from "react"
import { Button } from "@/components/ui/button"
import { AlertTriangle, RefreshCw } from "lucide-react"

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends React.Component<React.PropsWithChildren<{}>, ErrorBoundaryState> {
  constructor(props: React.PropsWithChildren<{}>) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by boundary:", error, errorInfo)
  }

  resetError = () => {
    this.setState({ hasError: false, error: undefined })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="flex items-center justify-center min-h-screen bg-cosmic-gradient text-white p-6 animate-pulse-slow"
          role="alert"
          aria-live="assertive"
          aria-label="An error has occurred. Please refresh or try again."
        >
          <div className="text-center max-w-md">
            <div className="mb-6">
              <AlertTriangle className="h-16 w-16 text-nebula-purple mx-auto mb-4 animate-pulse-slow" />
              <h2 className="text-2xl font-bold mb-2 text-logo-gradient">Something went wrong</h2>
              <p className="text-gray-400 mb-6">
                We encountered an unexpected error in the cosmic realm. Please try refreshing or attempting again.
              </p>
            </div>

            <div className="space-y-3">
              <Button
                onClick={() => window.location.reload()}
                className="btn-logo-gradient w-full glow"
                aria-label="Refresh the page to resolve the error"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh Page
              </Button>

              <Button
                variant="outline"
                onClick={this.resetError}
                className="w-full border-nebula-purple/30 text-nebula-purple hover:bg-nebula-purple/10"
                aria-label="Try again to resolve the error"
              >
                Try Again
              </Button>
            </div>

            {process.env.NODE_ENV === "development" && this.state.error && (
              <details className="mt-6 text-left">
                <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-300">
                  Error Details (Development)
                </summary>
                <pre className="mt-2 text-xs bg-gray-900 p-3 rounded overflow-auto text-red-400">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      )
    }

    return this.props.children
  }
}