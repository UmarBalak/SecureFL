import type { ReactNode } from "react"
import { X, Minus, Square } from "lucide-react"
import { cn } from "@/lib/utils"

interface TerminalProps {
  title: string
  children: ReactNode
  className?: string
}

export function Terminal({ title, children, className }: TerminalProps) {
  return (
    <div className={cn("border border-green-500/50 rounded-md overflow-hidden bg-black", className)}>
      <div className="flex items-center justify-between bg-green-900/30 px-4 py-2 border-b border-green-500/50">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="h-3 w-3 rounded-full bg-red-500" />
            <div className="h-3 w-3 rounded-full bg-yellow-500" />
            <div className="h-3 w-3 rounded-full bg-green-500" />
          </div>
          <h3 className="text-sm font-semibold">{title}</h3>
        </div>
        <div className="flex items-center gap-2">
          <Minus className="h-3 w-3 text-green-400" />
          <Square className="h-3 w-3 text-green-400" />
          <X className="h-3 w-3 text-green-400" />
        </div>
      </div>
      <div className="p-4 overflow-auto max-h-[70vh]">{children}</div>
    </div>
  )
}
