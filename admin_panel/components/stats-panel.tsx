"use client"

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import { Badge } from "@/components/ui/badge"
import { Cpu, Users, Database, Clock } from "lucide-react"

interface Client {
  contribution_count: number
  status: string
}

interface Model {
  version: number
  created_at: string
}

interface StatsPanelProps {
  clients: Client[]
  models: Model[]
  lastAggregation: string
}

export function StatsPanel({ clients, models, lastAggregation }: StatsPanelProps) {
  // Calculate statistics
  const totalClients = clients.length
  const activeClients = clients.filter((c) => c.status === "Active").length
  const totalContributions = clients.reduce((sum, client) => sum + client.contribution_count, 0)
  const latestModelVersion = models.length > 0 ? Math.max(...models.map((m) => m.version)) : 0

  // Data for contribution distribution chart
  const contributingClients = clients.filter((c) => c.contribution_count > 0).length
  const nonContributingClients = totalClients - contributingClients

  const pieData = [
    { name: "Contributing", value: contributingClients },
    { name: "Non-contributing", value: nonContributingClients },
  ]

  const COLORS = ["#22c55e", "#6b7280"]

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    if (!timestamp || timestamp === "N/A") return "N/A"

    // Format: YYYYMMDDHHMMSS
    const year = timestamp.substring(0, 4)
    const month = timestamp.substring(4, 6)
    const day = timestamp.substring(6, 8)
    const hour = timestamp.substring(8, 10)
    const minute = timestamp.substring(10, 12)
    const second = timestamp.substring(12, 14)

    return `${year}-${month}-${day} ${hour}:${minute}:${second}`
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-green-500/30 rounded-md p-4 bg-green-950/10">
          <div className="flex items-center gap-2 mb-2">
            <Users className="h-5 w-5 text-amber-400" />
            <h3 className="text-amber-400 font-semibold">Clients</h3>
          </div>
          <div className="text-2xl font-bold text-green-400">{totalClients}</div>
          <div className="text-xs text-green-600 mt-1">
            {activeClients} active / {totalClients - activeClients} inactive
          </div>
        </div>

        <div className="border border-green-500/30 rounded-md p-4 bg-green-950/10">
          <div className="flex items-center gap-2 mb-2">
            <Database className="h-5 w-5 text-amber-400" />
            <h3 className="text-amber-400 font-semibold">Models</h3>
          </div>
          <div className="text-2xl font-bold text-green-400">{models.length}</div>
          <div className="text-xs text-green-600 mt-1">Latest version: v{latestModelVersion}</div>
        </div>
      </div>

      <div className="border border-green-500/30 rounded-md p-4">
        <div className="flex items-center gap-2 mb-4">
          <Clock className="h-5 w-5 text-amber-400" />
          <h3 className="text-amber-400 font-semibold">System Timestamps</h3>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-green-600">Last Aggregation:</span>
            <span className="text-green-400">{formatTimestamp(lastAggregation)}</span>
          </div>

          <div className="flex justify-between">
            <span className="text-green-600">Latest Model Created:</span>
            <span className="text-green-400">
              {models.length > 0
                ? new Date(
                    models.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0]
                      .created_at,
                  ).toLocaleString()
                : "N/A"}
            </span>
          </div>

          <div className="flex justify-between">
            <span className="text-green-600">Total Contributions:</span>
            <span className="text-green-400">{totalContributions}</span>
          </div>
        </div>
      </div>

      <div className="border border-green-500/30 rounded-md p-4">
        <div className="flex items-center gap-2 mb-2">
          <Cpu className="h-5 w-5 text-amber-400" />
          <h3 className="text-amber-400 font-semibold">Client Contribution Distribution</h3>
        </div>

        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#22c55e",
                  border: "2px solid #ffffff",
                  borderRadius: "4px",
                  color: "#4ade80",
                }}
                formatter={(value) => [`${value} clients`, ""]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="flex justify-center gap-4 mt-2">
          <Badge className="bg-green-600 hover:bg-green-500">Contributing: {contributingClients}</Badge>
          <Badge className="bg-gray-500 hover:bg-gray-400">Non-contributing: {nonContributingClients}</Badge>
        </div>
      </div>
    </div>
  )
}
