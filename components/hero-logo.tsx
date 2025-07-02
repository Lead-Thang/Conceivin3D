import Image from "next/image"

export function HeroLogo({ size = 120 }: { size?: number }) {
  return (
    <div className="flex items-center justify-center mb-8">
      <div className="relative group">
        {/* Large logo container */}
        <div
          className="rounded-full flex items-center justify-center transition-all duration-500 group-hover:scale-105"
          style={{ width: `${size}px`, height: `${size}px` }}
        >
          <Image
            src="/logo.png"
            alt="Conceivin3D Logo"
            width={size}
            height={size}
            className="transition-all duration-500 group-hover:scale-105 drop-shadow-2xl"
            priority
          />
        </div>
        {/* Multiple animated glow rings that complement the logo */}
        <div className="absolute inset-0 bg-logo-purple/20 rounded-full blur-xl animate-pulse" />
        <div
          className="absolute inset-0 bg-logo-cyan/20 rounded-full blur-xl animate-pulse"
          style={{ animationDelay: "1s" }}
        />
        <div
          className="absolute inset-0 bg-gradient-to-br from-logo-purple/10 to-logo-cyan/10 rounded-full blur-2xl animate-pulse"
          style={{ animationDelay: "2s" }}
        />
      </div>
    </div>
  )
}
