"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Input } from "@/components/ui/input"
import { Search, Filter, ExternalLink } from "lucide-react"

interface Client {
  created_at: string
  contribution_count: number
  api_key: string
  csn: string
  client_id: string
  status: string
}

interface ClientsPanelProps {
  clients: Client[]
}

export function ClientsPanel({ clients }: ClientsPanelProps) {
  const router = useRouter()
  const [searchTerm, setSearchTerm] = useState("")
  const [sortField, setSortField] = useState<keyof Client>("created_at")
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc")

  const handleSort = (field: keyof Client) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  const filteredClients = clients
    .filter(
      (client) =>
        client.csn.toLowerCase().includes(searchTerm.toLowerCase()) ||
        client.client_id.toLowerCase().includes(searchTerm.toLowerCase()),
    )
    .sort((a, b) => {
      if (sortField === "contribution_count") {
        return sortDirection === "asc" ? a[sortField] - b[sortField] : b[sortField] - a[sortField]
      }

      if (sortField === "created_at") {
        return sortDirection === "asc"
          ? new Date(a[sortField]).getTime() - new Date(b[sortField]).getTime()
          : new Date(b[sortField]).getTime() - new Date(a[sortField]).getTime()
      }

      return sortDirection === "asc"
        ? a[sortField].localeCompare(b[sortField])
        : b[sortField].localeCompare(a[sortField])
    })

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-green-500" />
          <Input
            placeholder="Search clients by CSN or ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8 bg-black border-green-500/50 text-green-400 placeholder:text-green-700"
          />
        </div>
        <div className="flex items-center gap-1 text-xs">
          <Filter className="h-4 w-4" />
          <span>{clients.length} total clients</span>
        </div>
      </div>

      <div className="border border-green-500/30 rounded-md overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-green-950/30 border-green-500/30">
              <TableHead className="text-green-400 cursor-pointer" onClick={() => handleSort("csn")}>
                CSN {sortField === "csn" && (sortDirection === "asc" ? "↑" : "↓")}
              </TableHead>
              <TableHead className="text-green-400 cursor-pointer" onClick={() => handleSort("contribution_count")}>
                Contributions {sortField === "contribution_count" && (sortDirection === "asc" ? "↑" : "↓")}
              </TableHead>
              <TableHead className="text-green-400 cursor-pointer" onClick={() => handleSort("created_at")}>
                Created {sortField === "created_at" && (sortDirection === "asc" ? "↑" : "↓")}
              </TableHead>
              <TableHead className="text-green-400 cursor-pointer" onClick={() => handleSort("status")}>
                Status {sortField === "status" && (sortDirection === "asc" ? "↑" : "↓")}
              </TableHead>
              <TableHead className="text-green-400">API Key</TableHead>
              <TableHead className="text-green-400">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredClients.map((client) => (
              <TableRow key={client.client_id} className="hover:bg-green-950/30 border-green-500/30">
                <TableCell className="font-medium text-amber-400">{client.csn}</TableCell>
                <TableCell className={client.contribution_count > 0 ? "text-green-400" : "text-gray-500"}>
                  {client.contribution_count}
                </TableCell>
                <TableCell>{new Date(client.created_at).toLocaleString()}</TableCell>
                <TableCell>
                  <span
                    className={`px-2 py-1 rounded text-xs ${
                      client.status === "Active" ? "bg-green-900/50 text-green-400" : "bg-red-900/30 text-red-400"
                    }`}
                  >
                    {client.status}
                  </span>
                </TableCell>
                <TableCell className="font-mono text-xs text-gray-400">{client.api_key.substring(0, 8)}...</TableCell>
                <TableCell>
                  <button
                    onClick={() => router.push(`/clients/${client.client_id}`)}
                    className="flex items-center gap-1 text-green-400 hover:text-green-300"
                  >
                    <ExternalLink className="h-4 w-4" /> Details
                  </button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
