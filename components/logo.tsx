import Image from "next/image"
import Link from "next/link"

export function Logo() {
  return (
    <Link href="/" className="flex items-center space-x-3 group">
      <div className="relative">
        {/* Logo container with enhanced glow */}
        <div className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 group-hover:scale-110">
          <Image
            src="/logo.png"
            alt="Conceivin3D Logo"
            width={40}
            height={40}
            className="w-10 h-10 transition-all duration-300 group-hover:scale-105 drop-shadow-lg"
            priority
          />
        </div>
        {/* Enhanced glow effect that matches the logo colors */}
        <div className="absolute inset-0 w-10 h-10 bg-gradient-to-br from-logo-purple/30 to-logo-cyan/30 opacity-0 group-hover:opacity-60 blur-lg rounded-full transition-opacity duration-300" />
      </div>
      <span className="text-xl font-bold text-logo-gradient tracking-tight">Conceivin3D</span>
    </Link>
  )
}
