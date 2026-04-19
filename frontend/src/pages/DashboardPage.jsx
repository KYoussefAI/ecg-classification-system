import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from 'recharts'
import { LayoutDashboard, Activity, TrendingUp, Award, Loader2 } from 'lucide-react'
import { statsAPI } from '../utils/api'

const CLASS_COLORS = {
  NORM: '#22c55e',
  MI:   '#ef4444',
  CD:   '#f59e0b',
  HYP:  '#a855f7',
  STTC: '#3b82f6',
}

const MODEL_METRICS = [
  { cls: 'CD',   auc: 0.897, ap: 0.799, f1: 0.73 },
  { cls: 'HYP',  auc: 0.899, ap: 0.627, f1: 0.59 },
  { cls: 'MI',   auc: 0.927, ap: 0.834, f1: 0.74 },
  { cls: 'NORM', auc: 0.938, ap: 0.916, f1: 0.85 },
  { cls: 'STTC', auc: 0.926, ap: 0.802, f1: 0.75 },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card px-3 py-2 text-xs border border-blue-700/30">
      <p className="font-semibold text-white mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </p>
      ))}
    </div>
  )
}

export default function DashboardPage() {
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    statsAPI.get()
      .then(r  => setStats(r.data))
      .catch(() => setError('Could not load stats. Make sure you have predictions saved.'))
      .finally(() => setLoading(false))
  }, [])

  // Build pie data from stats
  const pieData = stats
    ? Object.entries(stats.class_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <LayoutDashboard className="w-7 h-7 text-blue-400" />
          Dashboard
        </h1>
        <p className="text-blue-400/60 mt-1">Model performance and your analysis statistics.</p>
      </div>

      {/* ── Model Performance Section ── */}
      <div className="mb-8">
        <p className="text-xs text-blue-400/60 uppercase tracking-widest font-semibold mb-4">
          Model Performance (PTB-XL Test Set)
        </p>

        {/* Summary cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Macro AUC',   value: '0.917', icon: Award,      color: 'text-blue-400'   },
            { label: 'Macro AP',    value: '0.796', icon: TrendingUp, color: 'text-purple-400' },
            { label: 'Macro F1',    value: '0.73',  icon: Activity,   color: 'text-green-400'  },
            { label: 'Test Records',value: '2,198', icon: Activity,   color: 'text-yellow-400' },
          ].map(m => (
            <div key={m.label} className="stat-card card-glow">
              <m.icon className={`w-5 h-5 ${m.color} mb-1`} />
              <p className="text-2xl font-bold text-white">{m.value}</p>
              <p className="text-xs text-blue-400/60">{m.label}</p>
            </div>
          ))}
        </div>

        {/* AUC / AP bar chart */}
        <div className="card card-glow">
          <p className="text-sm font-semibold text-white mb-4">Per-Class AUC & Average Precision</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={MODEL_METRICS} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a6e22" />
              <XAxis dataKey="cls" tick={{ fill: '#6b7db3', fontSize: 12 }} />
              <YAxis domain={[0.5, 1]} tick={{ fill: '#6b7db3', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#6b7db3' }} />
              <Bar dataKey="auc" name="ROC-AUC" fill="#3b82f6" radius={[4,4,0,0]} />
              <Bar dataKey="ap"  name="Avg Prec" fill="#8b5cf6" radius={[4,4,0,0]} />
              <Bar dataKey="f1"  name="F1 Score" fill="#22c55e" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── User Stats Section ── */}
      <div>
        <p className="text-xs text-blue-400/60 uppercase tracking-widest font-semibold mb-4">
          Your Analysis Stats
        </p>

        {loading && (
          <div className="flex items-center gap-3 text-blue-400/60 py-8 justify-center">
            <Loader2 className="w-5 h-5 animate-spin" />
            Loading your statistics…
          </div>
        )}

        {error && !loading && (
          <div className="card text-center py-8 text-blue-400/50">
            <Activity className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p>{error}</p>
            <p className="text-sm mt-1">Run your first analysis to see stats here.</p>
          </div>
        )}

        {stats && !loading && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Summary */}
            <div className="grid grid-cols-2 gap-4">
              <div className="stat-card card-glow">
                <p className="text-3xl font-bold text-white">{stats.total_predictions}</p>
                <p className="text-xs text-blue-400/60">Total Analyses</p>
              </div>
              <div className="stat-card card-glow">
                <p className="text-3xl font-bold text-white">
                  {Math.round(stats.avg_confidence * 100)}%
                </p>
                <p className="text-xs text-blue-400/60">Avg Confidence</p>
              </div>

              {/* Class distribution pie */}
              {pieData.length > 0 && (
                <div className="card card-glow col-span-2">
                  <p className="text-sm font-semibold text-white mb-3">Class Distribution</p>
                  <ResponsiveContainer width="100%" height={160}>
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name"
                           cx="50%" cy="50%" outerRadius={60} label={({ name, percent }) =>
                             `${name} ${(percent*100).toFixed(0)}%`
                           } labelLine={false}>
                        {pieData.map(entry => (
                          <Cell key={entry.name} fill={CLASS_COLORS[entry.name] || '#3b82f6'} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Activity trend */}
            {stats.recent_trend?.length > 0 ? (
              <div className="card card-glow">
                <p className="text-sm font-semibold text-white mb-4">Analysis Activity (last 30 days)</p>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={stats.recent_trend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a6e22" />
                    <XAxis dataKey="date" tick={{ fill: '#6b7db3', fontSize: 10 }}
                           tickFormatter={d => d.slice(5)} />
                    <YAxis allowDecimals={false} tick={{ fill: '#6b7db3', fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2}
                          dot={{ fill: '#3b82f6', r: 3 }} name="Analyses" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="card card-glow flex items-center justify-center text-blue-400/40 text-sm">
                No trend data yet. Run more analyses.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
