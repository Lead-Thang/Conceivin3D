"use client"

import { Button } from "./ui/button"
import { ArrowRight } from "lucide-react"
import { HeroLogo } from "./hero-logo"
import Link from "next/link"

export function Hero() {
  return (
    <div className="relative overflow-hidden bg-black">
      <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:60px_60px]" />

      {/* Animated background elements using logo colors */}
      <div className="absolute top-20 left-10 w-72 h-72 bg-logo-purple/20 rounded-full blur-3xl animate-pulse" />
      <div
        className="absolute bottom-20 right-10 w-96 h-96 bg-logo-cyan/20 rounded-full blur-3xl animate-pulse"
        style={{ animationDelay: "1s" }}
      />

      <div className="relative max-w-7xl mx-auto px-6 py-24 md:py-32">
        <div className="text-center max-w-3xl mx-auto">
          {/* Hero Logo */}
          <HeroLogo size={140} />

          <h1 className="text-4xl md:text-6xl font-bold mb-6 text-white">
            Design the Future with <span className="text-logo-gradient">Conceivin3D</span>
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-gray-300">
            The professional 3D design platform that brings CAD-like precision to the browser, powered by AI assistance.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/design">
              <Button size="lg" className="btn-logo-gradient text-white px-8 border-0">
                Start Designing
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Button
              variant="outline"
              size="lg"
              className="px-8 border-logo-purple/50 text-logo-purple hover:bg-logo-purple/10 bg-black/50"
            >
              Watch Demo
            </Button>
          </div>
        </div>
      </div>

      <div className="absolute -bottom-48 left-0 right-0 h-64 bg-black transform rotate-6 scale-125 z-10"></div>
    </div>
  )
}
