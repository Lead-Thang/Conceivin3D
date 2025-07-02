"use client"

import { CardWithImageLoader } from "@/components/ui/card-with-image-loader"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

export function FeatureCardGrid() {
  const router = useRouter()

  const features = [
    {
      title: "Describe",
      description: "Start by describing your product idea in natural language.",
      icon: "/placeholder.svg?height=128&width=128",
    },
    {
      title: "Generate",
      description: "AI creates a 3D model concept based on your prompt.",
      icon: "/placeholder.svg?height=128&width=128",
    },
    {
      title: "Design",
      description: "Refine and edit your design in real-time 3D viewer.",
      icon: "/placeholder.svg?height=128&width=128",
    },
  ]

  return (
    <section className="py-16 px-6 border-t border-logo-purple/20">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-2xl font-bold text-center mb-12 text-logo-gradient">
          From Concept to Creation in Minutes
        </h2>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <CardWithImageLoader
              key={index}
              src={feature.icon}
              alt={feature.title}
              title={feature.title}
              description={feature.description}
              isLoading={false}
              className="backdrop-blur-sm bg-gray-900/30 border border-logo-purple/20 shadow-lg hover:-translate-y-1 hover:shadow-xl hover:border-logo-cyan/40 transition-all duration-300"
            />
          ))}
        </div>
      </div>
    </section>
  )
}
