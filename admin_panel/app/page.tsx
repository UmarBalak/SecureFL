"use client"

import { useEffect, useState } from "react"
import { Terminal } from "@/components/terminal"
import { ClientsPanel } from "@/components/clients-panel"
import { ModelsPanel } from "@/components/models-panel"
import { StatsPanel } from "@/components/stats-panel"
import { fetchData } from "@/lib/api"
import { Loader2 } from "lucide-react"
import { useRouter } from "next/navigation"

export default function Dashboard() {
  const router = useRouter()
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const result = await fetchData()
        setData(result)
      } catch (err) {
        setError("Failed to fetch data. Check your API endpoint.")
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    loadData()
    // Set up polling every 30 seconds for real-time updates
    const interval = setInterval(loadData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-black text-green-500">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 animate-spin" />
          <div className="font-mono">Initializing system...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-black text-red-500 font-mono">
        <div className="border border-red-500 p-6 rounded-md">
          <h2 className="text-xl mb-2">$ ERROR</h2>
          <p>{error}</p>
          <p className="mt-4">Check console for details.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-green-500 p-4 font-mono">
      <header className="mb-6 border-b border-green-500/30 pb-2">
        <h1 className="text-2xl font-bold">
          <span className="text-amber-500">root@enspint</span>:
          <span className="text-blue-400">~/federated-learning</span>$ ./dashboard
        </h1>
        <p className="text-xs text-green-400">
          Last updated: {new Date().toLocaleString()} | Last system check:{" "}
          {formatTimestamp(data?.last_checked_timestamp)}
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Terminal title="Clients Status" className="md:col-span-2">
          <ClientsPanel clients={data?.clients || []} />
        </Terminal>

        <Terminal title="Global Models">
          <ModelsPanel models={data?.global_models || []} clients={data?.clients || []} />
        </Terminal>

        <Terminal title="System Statistics">
          <StatsPanel
            clients={data?.clients || []}
            models={data?.global_models || []}
            lastAggregation={data?.global_aggregation?.[0]?.value || "N/A"}
          />
        </Terminal>
      </div>
    </div>
  )
}

function formatTimestamp(timestamp: string) {
  if (!timestamp) return "Unknown"

  // Format: YYYYMMDDHHMMSS
  const year = timestamp.substring(0, 4)
  const month = timestamp.substring(4, 6)
  const day = timestamp.substring(6, 8)
  const hour = timestamp.substring(8, 10)
  const minute = timestamp.substring(10, 12)
  const second = timestamp.substring(12, 14)

  return `${year}-${month}-${day} ${hour}:${minute}:${second}`
}
