import { useEffect, useState } from 'react'
import { History, Trash2, Loader2, AlertCircle, Activity, Search } from 'lucide-react'
import { historyAPI } from '../utils/api'

const CLASS_COLORS = {
  NORM: 'text-green-400 bg-green-900/30 border-green-700/30',
  MI:   'text-red-400 bg-red-900/30 border-red-700/30',
  CD:   'text-yellow-400 bg-yellow-900/30 border-yellow-700/30',
  HYP:  'text-purple-400 bg-purple-900/30 border-purple-700/30',
  STTC: 'text-blue-400 bg-blue-900/30 border-blue-700/30',
}

function ConfBar({ value }) {
  const pct = Math.round(value * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 progress-bar">
        <div className={`progress-fill ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-white">{pct}%</span>
    </div>
  )
}

export default function HistoryPage() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')
  const [search,  setSearch]  = useState('')
  const [deleting,setDeleting]= useState(null)

  const load = () => {
    setLoading(true)
    historyAPI.getAll()
      .then(r => setRecords(r.data))
      .catch(() => setError('Failed to load history.'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (id) => {
    setDeleting(id)
    try {
      await historyAPI.remove(id)
      setRecords(r => r.filter(x => x.id !== id))
    } catch {
      setError('Delete failed.')
    } finally {
      setDeleting(null)
    }
  }

  const filtered = records.filter(r => {
    const q = search.toLowerCase()
    return !q
      || (r.patient_name || '').toLowerCase().includes(q)
      || r.top_class.toLowerCase().includes(q)
      || r.positive_classes.some(c => c.toLowerCase().includes(q))
  })

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <History className="w-7 h-7 text-blue-400" />
            Analysis History
          </h1>
          <p className="text-blue-400/60 mt-1">
            All your past ECG analyses — {records.length} record{records.length !== 1 ? 's' : ''}.
          </p>
        </div>
      </div>

      {/* Search */}
      <div className="relative mb-6 max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-blue-500/50" />
        <input
          className="input pl-9 text-sm"
          placeholder="Search by patient, class…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* States */}
      {loading && (
        <div className="flex items-center gap-3 justify-center py-16 text-blue-400/60">
          <Loader2 className="w-5 h-5 animate-spin" />
          Loading history…
        </div>
      )}

      {error && !loading && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-900/20 border border-red-700/30 text-red-300 text-sm">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="card text-center py-16 text-blue-400/40">
          <Activity className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="font-medium">{search ? 'No matching records.' : 'No analyses yet.'}</p>
          <p className="text-sm mt-1">Run your first ECG analysis to see it here.</p>
        </div>
      )}

      {!loading && filtered.length > 0 && (
        <div className="card card-glow overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-blue-900/30 bg-[#080e1c]">
                  {['Date', 'Patient', 'Age/Sex', 'Top Class', 'Confidence', 'Findings', 'Actions'].map(h => (
                    <th key={h} className="text-left px-4 py-3 text-xs font-semibold text-blue-400/60 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={r.id}
                      className={`border-b border-blue-900/20 hover:bg-blue-900/10 transition-colors
                                  ${i % 2 === 0 ? '' : 'bg-blue-950/10'}`}>

                    <td className="px-4 py-3 text-blue-400/60 text-xs font-mono whitespace-nowrap">
                      {r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : '—'}
                    </td>

                    <td className="px-4 py-3">
                      <span className="font-medium text-white">{r.patient_name || <span className="text-blue-600/40 italic">anonymous</span>}</span>
                    </td>

                    <td className="px-4 py-3 text-blue-300/60 text-xs">
                      {r.age || '—'} {r.sex || ''}
                    </td>

                    <td className="px-4 py-3">
                      <span className={`badge border ${CLASS_COLORS[r.top_class] || 'text-blue-300 bg-blue-900/30 border-blue-700/30'}`}>
                        {r.top_class}
                      </span>
                    </td>

                    <td className="px-4 py-3">
                      <ConfBar value={r.confidence} />
                    </td>

                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {(r.positive_classes || []).map(c => (
                          <span key={c} className={`badge border text-[10px] ${CLASS_COLORS[c] || 'text-blue-300 bg-blue-900/30 border-blue-700/30'}`}>
                            {c}
                          </span>
                        ))}
                      </div>
                    </td>

                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleDelete(r.id)}
                        disabled={deleting === r.id}
                        className="p-1.5 rounded-lg text-red-400/50 hover:text-red-300 hover:bg-red-900/20
                                   transition-colors disabled:opacity-30"
                      >
                        {deleting === r.id
                          ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          : <Trash2 className="w-3.5 h-3.5" />
                        }
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
