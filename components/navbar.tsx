"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import { Button } from "./ui/button"
import { Menu, X, Lightbulb, CuboidIcon as Cube, Layers } from "lucide-react"
import { Logo } from "./logo"
import { ThemeToggle } from "./theme-toggle"

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="bg-black/95 backdrop-blur-sm border-b border-logo-purple/20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left: Logo - Logo component already contains Link */}
          <div className="flex items-center">
            <Logo />
          </div>

          {/* Center: Desktop Navigation */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <NavLink href="/" icon={<Lightbulb className="h-4 w-4 mr-2" />}>
                Home
              </NavLink>
              <NavLink href="/design" icon={<Cube className="h-4 w-4 mr-2" />}>
                Design
              </NavLink>
              <NavLink href="#features" icon={<Layers className="h-4 w-4 mr-2" />}>
                Features
              </NavLink>
              {/* Free Forever Badge */}
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-logo-cyan/20 text-logo-cyan">
                Free Forever
              </span>
            </div>
          </div>

          {/* Right: Actions */}
          <div className="hidden md:flex items-center space-x-4">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="sm"
              className="text-gray-300 hover:text-logo-purple hover:bg-logo-purple/10 transition-all duration-300"
            >
              Sign In
            </Button>
            <Button size="sm" className="btn-logo-gradient text-white shadow-lg hover:shadow-logo-purple/30 border-0">
              Get Started
            </Button>
          </div>

          {/* Mobile Menu Toggle */}
          <div className="md:hidden flex items-center space-x-2">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(!isOpen)}
              aria-label={isOpen ? "Close navigation" : "Open navigation"}
              className="text-gray-300 hover:text-logo-purple focus:outline-none"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden border-t border-logo-purple/20">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-black/95 backdrop-blur-md">
            <MobileLink href="/" label="Home" onClick={() => setIsOpen(false)} />
            <MobileLink href="/design" label="Design" onClick={() => setIsOpen(false)} />
            <MobileLink href="#features" label="Features" onClick={() => setIsOpen(false)} />
            {/* Mobile Badge */}
            <div className="px-3 py-2">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-logo-cyan/20 text-logo-cyan">
                Free Forever
              </span>
            </div>
          </div>
        </div>
      )}
    </nav>
  )
}

// Reusable nav link component
function NavLink({ href, children, icon }: { href: string; children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="text-gray-300 hover:text-logo-purple px-3 py-2 rounded-md text-sm font-medium flex items-center transition-all duration-300 hover:bg-logo-purple/10"
    >
      {icon}
      {children}
    </Link>
  )
}

// Mobile-friendly link component
function MobileLink({
  href,
  label,
  onClick,
}: {
  href: string
  label: string
  onClick?: () => void
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className="block px-3 py-2 rounded-md text-base font-medium text-gray-300 hover:text-logo-cyan hover:bg-logo-cyan/10 transition-all duration-300"
    >
      {label}
    </Link>
  )
}
