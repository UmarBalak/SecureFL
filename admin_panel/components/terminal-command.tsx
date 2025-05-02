"use client"

import { useState, type KeyboardEvent } from "react"
import { ArrowRight } from "lucide-react"

interface TerminalCommandProps {
  onCommand: (command: string) => void
}

export function TerminalCommand({ onCommand }: TerminalCommandProps) {
  const [command, setCommand] = useState("")

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && command.trim()) {
      onCommand(command)
      setCommand("")
    }
  }

  return (
    <div className="flex items-center gap-2 border border-green-500/30 rounded-md p-2 bg-black">
      <span className="text-amber-500">root@harry:~$</span>
      <input
        type="text"
        value={command}
        onChange={(e) => setCommand(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-1 bg-transparent border-none outline-none text-green-400"
        placeholder="Type a command..."
      />
      <button
        onClick={() => {
          if (command.trim()) {
            onCommand(command)
            setCommand("")
          }
        }}
        className="text-green-500 hover:text-green-400"
      >
        <ArrowRight className="h-4 w-4" />
      </button>
    </div>
  )
}
