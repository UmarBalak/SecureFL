"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { Terminal } from "@/components/terminal"
import { fetchData } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Clock, Database, Key, Server, Share2 } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function ClientDetailsPage() {
  const params = useParams()
  const router = useRouter()
  const clientId = params.clientId as string

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
  }, [])

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-black text-green-500">
        <div className="font-mono">Loading client data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-black text-red-500 font-mono">
        <div className="border border-red-500 p-6 rounded-md">
          <h2 className="text-xl mb-2">$ ERROR</h2>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  const client = data?.clients.find((c: any) => c.client_id === clientId)

  if (!client) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-black text-red-500 font-mono">
        <div className="border border-red-500 p-6 rounded-md">
          <h2 className="text-xl mb-2">$ CLIENT_NOT_FOUND</h2>
          <p>Client with ID {clientId} not found.</p>
          <Button
            variant="outline"
            className="mt-4 border-green-500 text-green-500 hover:bg-green-950"
            onClick={() => router.push("/")}
          >
            <ArrowLeft className="mr-2 h-4 w-4" /> Return to Dashboard
          </Button>
        </div>
      </div>
    )
  }

  // Find models this client contributed to
  const contributedModels =
    data?.global_models.filter((model: any) => model.client_ids.split(",").includes(client.client_id)) || []

  // Prepare chart data for contribution timeline
  const contributionData = contributedModels.map((model: any) => ({
    name: `v${model.version}`,
    date: new Date(model.created_at).toLocaleDateString(),
    value: 1,
  }))

  return (
    <div className="min-h-screen bg-black text-green-500 p-4 font-mono">
      <header className="mb-6 border-b border-green-500/30 pb-2">
        <div className="flex items-center">
          <Button
            variant="outline"
            className="mr-4 border-green-500 text-green-500 hover:bg-green-950"
            onClick={() => router.push("/")}
          >
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>
          <div>
            <h1 className="text-2xl font-bold">
              <span className="text-amber-500">Client:</span> {client.csn}
            </h1>
            <p className="text-xs text-green-400">ID: {client.client_id}</p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="border border-green-500/30 rounded-md p-4 bg-green-950/10">
          <div className="flex items-center gap-2 mb-2">
            <Server className="h-5 w-5 text-amber-400" />
            <h3 className="text-amber-400 font-semibold">Status</h3>
          </div>
          <Badge className={client.status === "Active" ? "bg-green-700" : "bg-red-900"}>{client.status}</Badge>
        </div>

        <div className="border border-green-500/30 rounded-md p-4 bg-green-950/10">
          <div className="flex items-center gap-2 mb-2">
            <Share2 className="h-5 w-5 text-amber-400" />
            <h3 className="text-amber-400 font-semibold">Contributions</h3>
          </div>
          <div className="text-2xl font-bold text-green-400">{client.contribution_count}</div>
        </div>

        <div className="border border-green-500/30 rounded-md p-4 bg-green-950/10">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="h-5 w-5 text-amber-400" />
            <h3 className="text-amber-400 font-semibold">Created</h3>
          </div>
          <div className="text-sm text-green-400">{new Date(client.created_at).toLocaleString()}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Terminal title="Client Details">
          <div className="space-y-4">
            <div className="border border-green-500/30 rounded-md p-4 bg-black">
              <div className="grid grid-cols-1 gap-2">
                <div className="flex items-center gap-2">
                  <Key className="h-4 w-4 text-amber-400" />
                  <span className="text-green-600">API Key:</span>
                  <code className="bg-green-950/20 px-2 py-1 rounded text-amber-400 text-xs">{client.api_key}</code>
                </div>
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-amber-400" />
                  <span className="text-green-600">CSN:</span>
                  <span className="text-green-400">{client.csn}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Server className="h-4 w-4 text-amber-400" />
                  <span className="text-green-600">Client ID:</span>
                  <span className="text-green-400">{client.client_id}</span>
                </div>
              </div>
            </div>

            <div className="border-t border-green-500/30 pt-4">
              <h3 className="text-amber-400 font-semibold mb-2">System Commands</h3>
              <div className="grid grid-cols-1 gap-2">
                <div className="bg-green-950/10 p-2 rounded-md">
                  <code className="text-xs">$ ./activate_client --id={client.client_id}</code>
                </div>
                <div className="bg-green-950/10 p-2 rounded-md">
                  <code className="text-xs">$ ./reset_contributions --csn={client.csn}</code>
                </div>
                <div className="bg-green-950/10 p-2 rounded-md">
                  <code className="text-xs">$ ./regenerate_api_key --id={client.client_id}</code>
                </div>
              </div>
            </div>
          </div>
        </Terminal>

        <Terminal title="Contribution History">
          {client.contribution_count > 0 ? (
            <div className="space-y-4">
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={contributionData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
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
                    <Bar dataKey="value" fill="#22c55e">
                      {contributionData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={`#22c55e`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="text-xs text-center text-green-400">Contribution Timeline (Model Versions)</div>

              <div className="border-t border-green-500/30 pt-4">
                <h3 className="text-amber-400 font-semibold mb-2">Contributed Models</h3>
                <div className="border border-green-500/30 rounded-md overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="hover:bg-green-950/30 border-green-500/30">
                        <TableHead className="text-green-400">Version</TableHead>
                        <TableHead className="text-green-400">Created At</TableHead>
                        <TableHead className="text-green-400">Contributors</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {contributedModels.map((model: any) => (
                        <TableRow key={model.id} className="hover:bg-green-950/30 border-green-500/30">
                          <TableCell className="font-medium text-amber-400">v{model.version}</TableCell>
                          <TableCell>{new Date(model.created_at).toLocaleString()}</TableCell>
                          <TableCell>{model.num_clients_contributed}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full py-8">
              <div className="text-amber-400 mb-2">No contributions yet</div>
              <div className="text-xs text-green-600 text-center max-w-xs">
                This client has not contributed to any global models. Contributions will appear here once the client
                participates in model training.
              </div>
            </div>
          )}
        </Terminal>
      </div>
    </div>
  )
}
