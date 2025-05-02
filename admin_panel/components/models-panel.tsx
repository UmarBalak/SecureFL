"use client"

import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChevronDown, ChevronUp } from "lucide-react"
import React from "react"

interface Model {
  num_clients_contributed: number
  version: number
  id: number
  created_at: string
  client_ids: string
}

interface Client {
  client_id: string
  csn: string
  contribution_count: number
}

interface ModelsPanelProps {
  models: Model[]
  clients: Client[]
}

export function ModelsPanel({ models, clients }: ModelsPanelProps) {
  const [expandedModel, setExpandedModel] = useState<number | null>(null)

  const toggleExpand = (modelId: number) => {
    setExpandedModel(expandedModel === modelId ? null : modelId)
  }

  const getClientName = (clientId: string) => {
    const client = clients.find((c) => c.client_id === clientId)
    return client ? client.csn : "Unknown"
  }

  const chartData = models.map((model) => ({
    name: `v${model.version}`,
    contributors: model.num_clients_contributed,
    id: model.id,
  }))

  return (
    <Tabs defaultValue="chart">
      <TabsList className="bg-green-900/20 border border-green-500/30">
        <TabsTrigger value="chart" className="data-[state=active]:bg-green-900/40">
          Chart View
        </TabsTrigger>
        <TabsTrigger value="table" className="data-[state=active]:bg-green-900/40">
          Table View
        </TabsTrigger>
      </TabsList>

      <TabsContent value="chart" className="mt-4">
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <XAxis dataKey="name" stroke="#4ade80" />
              <YAxis stroke="#4ade80" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #22c55e",
                  borderRadius: "4px",
                  color: "#4ade80",
                }}
                labelStyle={{ color: "#4ade80" }}
              />
              <Bar dataKey="contributors" fill="#22c55e">
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={`#22c55e`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="text-xs text-center mt-2 text-green-400">
          Global Model Versions and Number of Contributing Clients
        </div>
      </TabsContent>

      <TabsContent value="table" className="mt-4">
        <div className="border border-green-500/30 rounded-md overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-green-950/30 border-green-500/30">
                <TableHead className="text-green-400">Version</TableHead>
                <TableHead className="text-green-400">Created At</TableHead>
                <TableHead className="text-green-400">Contributors</TableHead>
                <TableHead className="text-green-400">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
  {models.map((model) => (
    <React.Fragment key={model.id}>
      <TableRow className="hover:bg-green-950/30 border-green-500/30">
        <TableCell className="font-medium text-amber-400">v{model.version}</TableCell>
        <TableCell>{new Date(model.created_at).toLocaleString()}</TableCell>
        <TableCell>{model.num_clients_contributed}</TableCell>
        <TableCell>
          <button
            onClick={() => toggleExpand(model.id)}
            className="flex items-center gap-1 text-green-400 hover:text-green-300"
          >
            {expandedModel === model.id ? (
              <>
                <ChevronUp className="h-4 w-4" /> Hide
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4" /> Show
              </>
            )}
          </button>
        </TableCell>
      </TableRow>
      {expandedModel === model.id && (
        <TableRow className="bg-green-950/20 border-green-500/30">
          <TableCell colSpan={4} className="p-2">
            <div className="text-xs p-2 border border-green-500/30 rounded bg-black">
              <div className="font-bold mb-1 text-green-400">Contributing Clients:</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
                {Array.from(new Set(model.client_ids.split(","))).map((clientId) => (
                  <div key={clientId} className="flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-green-500"></span>
                    <span className="text-amber-400">{getClientName(clientId)}</span>
                    <span className="text-gray-400 text-xs">({clientId.substring(0, 8)}...)</span>
                  </div>
                ))}
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}
    </React.Fragment>
  ))}
</TableBody>
          </Table>
        </div>
      </TabsContent>
    </Tabs>
  )
}
