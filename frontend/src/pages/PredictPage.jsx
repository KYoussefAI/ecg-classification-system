import { useState, useRef } from 'react'
import {
  Upload, Zap, Activity, User, AlertCircle,
  Loader2, FileUp, RefreshCw, Info
} from 'lucide-react'
import { predictAPI } from '../utils/api'
import ResultCard from '../components/ResultCard'

/* ── Generate a synthetic ECG-like signal for demo purposes ── */
function generateDemoSignal() {
  const N    = 1000
  const LEADS = 12
  const signal = []

  for (let t = 0; t < N; t++) {
    const row = []
    for (let l = 0; l < LEADS; l++) {
      // Simple synthetic QRS-like waveform
      const phase   = (t / N) * Math.PI * 2 * 10           // 10 beats
      const qrs     = Math.exp(-0.5 * Math.pow(((t % 100) - 50) / 5, 2)) * 0.8
      const p_wave  = Math.exp(-0.5 * Math.pow(((t % 100) - 30) / 8, 2)) * 0.15
      const t_wave  = Math.exp(-0.5 * Math.pow(((t % 100) - 70) / 12, 2)) * 0.2
      const noise   = (Math.random() - 0.5) * 0.02
      const lead_var = (l % 3 - 1) * 0.1
      row.push(parseFloat((qrs + p_wave + t_wave + noise + lead_var).toFixed(4)))
    }
    signal.push(row)
  }
  return signal
}

export default function PredictPage() {
  const [form, setForm] = useState({
    patientName: '',
    age:         '',
    sex:         '',
  })
  const [signal,  setSignal]  = useState(null)
  const [fileName, setFileName] = useState('')
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const fileRef = useRef()

  /* Parse JSON: raw 2-D array, or { "signal_data": [...] } (export style) */
  const handleFile = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setFileName(file.name)
    setError('')
    setResult(null)

    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const rows = Array.isArray(data) ? data : data?.signal_data
      if (!rows || !Array.isArray(rows)) {
        setError('JSON must be a 2-D array or an object with a "signal_data" array.')
        setSignal(null)
        return
      }
      setSignal(rows)
    } catch {
      setError('Could not parse file. Please upload a JSON file containing the signal array.')
      setSignal(null)
    }
  }

  const handleDemo = () => {
    setSignal(generateDemoSignal())
    setFileName('demo_signal.json')
    setResult(null)
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!signal) {
      setError('Please upload a signal or use the demo.')
      return
    }

    setError('')
    setLoading(true)
    setResult(null)

    const ageNum = form.age === '' ? null : Number(form.age)
    const payload = {
      signal_data: signal,
      ...(form.patientName.trim() ? { patient_name: form.patientName.trim() } : {}),
      ...(ageNum !== null && !Number.isNaN(ageNum) ? { age: ageNum } : {}),
      ...(form.sex ? { sex: form.sex } : {}),
    }

    try {
      const { data } = await predictAPI.predict(payload)
      setResult(data)
    } catch (err) {
      const d = err.response?.data?.detail
      if (typeof d === 'string') {
        setError(d)
      } else if (Array.isArray(d)) {
        setError(d.map((x) => x.msg || JSON.stringify(x)).join(' · '))
      } else {
        setError(err.message || 'Prediction failed. Is the API running?')
      }
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="p-8 max-w-3xl mx-auto animate-fade-in">

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <Activity className="w-7 h-7 text-blue-400" />
          ECG Analysis
        </h1>
        <p className="text-blue-400/60 mt-1">
          Upload a 12-lead ECG signal (JSON format) for AI-powered cardiac classification.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* ── Signal Upload ── */}
        <div className="card card-glow">
          <p className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <FileUp className="w-4 h-4 text-blue-400" />
            ECG Signal
          </p>

          {/* Drop zone */}
          <div
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-blue-800/50 rounded-xl p-8 text-center cursor-pointer
                       hover:border-blue-600/60 hover:bg-blue-900/10 transition-all duration-200"
          >
            <Upload className="w-8 h-8 text-blue-500 mx-auto mb-3" />
            {fileName ? (
              <p className="text-blue-300 font-medium">{fileName}</p>
            ) : (
              <>
                <p className="text-white font-medium mb-1">Drop your JSON signal here</p>
                <p className="text-blue-400/50 text-sm">Shape: (1000, 12) or (12, 1000)</p>
              </>
            )}
            <input ref={fileRef} type="file" accept=".json,.npy" onChange={handleFile} className="hidden" />
          </div>

          {/* Demo button */}
          <div className="flex items-center gap-3 mt-4">
            <div className="flex-1 border-t border-blue-900/30" />
            <span className="text-xs text-blue-600/50">or</span>
            <div className="flex-1 border-t border-blue-900/30" />
          </div>
          <button type="button" onClick={handleDemo}
                  className="btn-ghost w-full mt-3 border border-blue-800/30 rounded-xl py-2.5 text-sm">
            <Zap className="w-4 h-4" />
            Use Demo Signal
          </button>

          {/* Info */}
          <div className="flex items-start gap-2 mt-4 px-3 py-2.5 rounded-lg bg-blue-900/20 text-xs text-blue-400/70">
            <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <span>
              JSON file must contain a 2-D array of shape <code className="font-mono">[1000][12]</code> (1000 timesteps × 12 leads at 100 Hz).
              Export from PTB-XL using <code className="font-mono">np.save</code> then convert with Python's <code className="font-mono">json.dumps(arr.tolist())</code>.
            </span>
          </div>
        </div>

        {/* ── Patient Info (optional) ── */}
        <div className="card card-glow">
          <p className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <User className="w-4 h-4 text-blue-400" />
            Patient Information <span className="text-blue-600/50 font-normal">(optional)</span>
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="sm:col-span-3">
              <label className="label">Patient Name</label>
              <input className="input" placeholder="e.g. John Doe" value={form.patientName}
                onChange={e => setForm(f => ({ ...f, patientName: e.target.value }))} />
            </div>
            <div>
              <label className="label">Age</label>
              <input className="input" type="number" placeholder="e.g. 55" min="0" max="130"
                value={form.age} onChange={e => setForm(f => ({ ...f, age: e.target.value }))} />
            </div>
            <div>
              <label className="label">Sex</label>
              <select className="input" value={form.sex}
                onChange={e => setForm(f => ({ ...f, sex: e.target.value }))}>
                <option value="">— select —</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-900/20 border border-red-700/30 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Submit */}
        <button type="submit" className="btn-primary w-full py-4 text-base" disabled={loading}>
          {loading
            ? <><Loader2 className="w-5 h-5 animate-spin" /> Analysing ECG…</>
            : <><Activity className="w-5 h-5" /> Run Analysis</>
          }
        </button>
      </form>

      {/* ── Result ── */}
      {result && (
        <div className="mt-8">
          <ResultCard result={result} />
          <button onClick={() => { setResult(null); setSignal(null); setFileName('') }}
                  className="btn-ghost mt-4 text-sm">
            <RefreshCw className="w-4 h-4" /> New Analysis
          </button>
        </div>
      )}
    </div>
  )
}
