import { CheckCircle, AlertTriangle, Activity } from 'lucide-react'

const CLASS_META = {
  NORM: { color: 'green',  label: 'Normal ECG',              icon: CheckCircle    },
  MI:   { color: 'red',    label: 'Myocardial Infarction',   icon: AlertTriangle  },
  CD:   { color: 'yellow', label: 'Conduction Disturbance',  icon: Activity       },
  HYP:  { color: 'purple', label: 'Hypertrophy',             icon: Activity       },
  STTC: { color: 'blue',   label: 'ST/T-Change',             icon: Activity       },
}

const COLOR_MAP = {
  green:  { bar: 'bg-green-500',  text: 'text-green-400',  badge: 'bg-green-900/30 text-green-300 border-green-700/30'  },
  red:    { bar: 'bg-red-500',    text: 'text-red-400',    badge: 'bg-red-900/30 text-red-300 border-red-700/30'        },
  yellow: { bar: 'bg-yellow-500', text: 'text-yellow-400', badge: 'bg-yellow-900/30 text-yellow-300 border-yellow-700/30'},
  purple: { bar: 'bg-purple-500', text: 'text-purple-400', badge: 'bg-purple-900/30 text-purple-300 border-purple-700/30'},
  blue:   { bar: 'bg-blue-500',   text: 'text-blue-400',   badge: 'bg-blue-900/30 text-blue-300 border-blue-700/30'    },
}

function ProbBar({ cls, prob, positive }) {
  const meta  = CLASS_META[cls]  || { color: 'blue', label: cls }
  const c     = COLOR_MAP[meta.color]
  const pct   = Math.round(prob * 100)

  return (
    <div className="flex items-center gap-3">
      <div className="w-14 text-right">
        <span className={`text-xs font-mono font-bold ${c.text}`}>{cls}</span>
      </div>
      <div className="flex-1 progress-bar">
        <div className={`progress-fill ${c.bar}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="w-12 text-right">
        <span className="text-sm font-mono text-white">{pct}%</span>
      </div>
      {positive && (
        <span className={`badge border ${c.badge} text-[10px]`}>+</span>
      )}
    </div>
  )
}

export default function ResultCard({ result }) {
  if (!result) return null

  const { predictions, top_class, confidence, positive_classes, patient_name, age, sex } = result
  const isNormal     = positive_classes.includes('NORM') && positive_classes.length === 1
  const hasAbnormal  = positive_classes.some(c => c !== 'NORM')

  return (
    <div className="card card-glow animate-slide-up space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-blue-400/60 uppercase tracking-wider mb-1">Analysis Result</p>
          {patient_name && (
            <p className="text-sm text-blue-300/70 mb-0.5">
              Patient: <span className="text-white font-medium">{patient_name}</span>
              {age && <span className="ml-2 text-blue-400/50">· {age}y</span>}
              {sex && <span className="ml-1 text-blue-400/50">· {sex}</span>}
            </p>
          )}
          <h2 className="text-xl font-bold text-white">
            {isNormal ? '✓ Normal ECG' : hasAbnormal ? '⚠ Abnormal Findings' : 'Analysis Complete'}
          </h2>
        </div>
        <div className={`px-3 py-1.5 rounded-xl text-sm font-semibold ${
          isNormal ? 'bg-green-900/30 text-green-300 border border-green-700/30'
                   : 'bg-red-900/30 text-red-300 border border-red-700/30'
        }`}>
          {isNormal ? 'Normal' : 'Review'}
        </div>
      </div>

      {/* Positive classes */}
      {positive_classes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {positive_classes.map(cls => {
            const meta = CLASS_META[cls] || { color: 'blue', label: cls }
            const c    = COLOR_MAP[meta.color]
            return (
              <span key={cls} className={`badge border ${c.badge}`}>
                {meta.label || cls}
              </span>
            )
          })}
        </div>
      )}

      {/* Probability bars */}
      <div>
        <p className="text-xs text-blue-400/60 uppercase tracking-wider mb-3">Class Probabilities</p>
        <div className="space-y-3">
          {predictions.map(p => (
            <ProbBar key={p.class} cls={p.class} prob={p.probability} positive={p.positive} />
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="pt-3 border-t border-blue-900/30 flex items-center justify-between text-xs text-blue-400/50">
        <span>Top prediction: <span className="text-blue-300 font-mono font-medium">{top_class}</span></span>
        <span>Confidence: <span className="text-blue-300 font-mono font-medium">{Math.round(confidence * 100)}%</span></span>
      </div>
    </div>
  )
}
