import { Link } from 'react-router-dom'
import { Heart, Activity, Shield, Zap, BarChart2, ArrowRight, CheckCircle } from 'lucide-react'

const CLASSES = [
  { code: 'NORM', label: 'Normal',                  color: 'text-green-400',  bg: 'bg-green-900/20' },
  { code: 'MI',   label: 'Myocardial Infarction',   color: 'text-red-400',    bg: 'bg-red-900/20'   },
  { code: 'CD',   label: 'Conduction Disturbance',  color: 'text-yellow-400', bg: 'bg-yellow-900/20'},
  { code: 'HYP',  label: 'Hypertrophy',             color: 'text-purple-400', bg: 'bg-purple-900/20'},
  { code: 'STTC', label: 'ST/T-Change',             color: 'text-blue-400',   bg: 'bg-blue-900/20'  },
]

const FEATURES = [
  { icon: Zap,      title: 'Real-time Inference',  desc: 'Sub-second analysis of 12-lead ECG signals powered by a residual neural network.' },
  { icon: Shield,   title: 'Multi-label Detection',desc: 'Simultaneously detects 5 cardiac superclasses — not just binary healthy/sick.' },
  { icon: BarChart2,title: 'Threshold-Optimised',  desc: 'Per-class thresholds tuned on PTB-XL test fold for best F1 on each condition.' },
]

const METRICS = [
  { label: 'Macro AUC',   value: '0.917' },
  { label: 'MI AUC',      value: '0.927' },
  { label: 'NORM AUC',    value: '0.938' },
  { label: 'Test Samples',value: '2,198' },
]

/* ── Animated ECG SVG ── */
function EcgHero() {
  return (
    <svg viewBox="0 0 600 120" className="w-full opacity-60" fill="none"
         xmlns="http://www.w3.org/2000/svg">
      <path
        d="M0 60 L80 60 L100 60 L110 20 L120 100 L130 10 L145 90 L160 60 L220 60
           L240 60 L250 20 L260 100 L270 10 L285 90 L300 60 L360 60
           L380 60 L390 20 L400 100 L410 10 L425 90 L440 60 L600 60"
        stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        className="ecg-animate"
      />
    </svg>
  )
}

export default function HomePage() {
  return (
    <div className="min-h-screen ecg-grid text-[#e8f0fe] overflow-x-hidden">

      {/* ── Nav ── */}
      <nav className="flex items-center justify-between px-8 py-5 border-b border-blue-900/30 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center"
               style={{ boxShadow: '0 0 20px rgba(59,130,246,0.4)' }}>
            <Heart className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-xl">CardioScan</span>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/login"    className="btn-ghost text-sm">Sign In</Link>
          <Link to="/register" className="btn-primary text-sm px-5 py-2.5">Get Started</Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="max-w-7xl mx-auto px-8 pt-24 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-blue-500/30
                        bg-blue-900/20 text-blue-300 text-xs font-medium mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          PTB-XL Dataset · 21,799 ECG Records · 5 Cardiac Classes
        </div>

        <h1 className="text-6xl font-bold leading-tight mb-6 max-w-4xl mx-auto">
          AI-Powered{' '}
          <span className="text-transparent bg-clip-text"
                style={{ backgroundImage: 'linear-gradient(135deg, #3b82f6, #60a5fa)' }}>
            ECG Analysis
          </span>
          {' '}in Seconds
        </h1>
        <p className="text-xl text-blue-300/70 max-w-2xl mx-auto mb-10">
          Upload a 12-lead ECG signal and get instant multi-label cardiac classification
          using a deep residual neural network trained on real clinical data.
        </p>

        <div className="flex items-center justify-center gap-4">
          <Link to="/register" className="btn-primary text-base px-8 py-4">
            Try Free Analysis <ArrowRight className="w-5 h-5" />
          </Link>
          <Link to="/login" className="btn-ghost text-base px-8 py-4 border border-blue-800/40 rounded-xl">
            Sign In
          </Link>
        </div>

        {/* ECG wave */}
        <div className="mt-16 max-w-2xl mx-auto">
          <EcgHero />
        </div>
      </section>

      {/* ── Metrics strip ── */}
      <section className="border-y border-blue-900/30 py-8 bg-blue-950/20">
        <div className="max-w-7xl mx-auto px-8 grid grid-cols-2 md:grid-cols-4 gap-6">
          {METRICS.map((m) => (
            <div key={m.label} className="text-center">
              <p className="text-3xl font-bold text-white">{m.value}</p>
              <p className="text-sm text-blue-400/70 mt-1">{m.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Detectable classes ── */}
      <section className="max-w-7xl mx-auto px-8 py-20">
        <p className="text-center text-blue-400/60 text-sm uppercase tracking-widest font-semibold mb-3">
          Detectable Conditions
        </p>
        <h2 className="text-3xl font-bold text-center mb-10">5 Cardiac Superclasses</h2>
        <div className="flex flex-wrap justify-center gap-3">
          {CLASSES.map((c) => (
            <div key={c.code}
                 className={`flex items-center gap-2.5 px-5 py-3 rounded-xl ${c.bg} border border-white/5`}>
              <Activity className={`w-4 h-4 ${c.color}`} />
              <div>
                <span className={`text-xs font-mono font-bold ${c.color}`}>{c.code}</span>
                <span className="text-sm text-white/70 ml-2">{c.label}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section className="max-w-7xl mx-auto px-8 pb-20">
        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map((f) => (
            <div key={f.title} className="card card-glow hover:border-blue-600/30 transition-colors">
              <div className="w-10 h-10 rounded-xl bg-blue-600/20 flex items-center justify-center mb-4">
                <f.icon className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">{f.title}</h3>
              <p className="text-sm text-blue-300/60 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="max-w-7xl mx-auto px-8 pb-24 text-center">
        <div className="card card-glow max-w-2xl mx-auto py-12"
             style={{ background: 'linear-gradient(135deg, #0d1529 0%, #0a1a3a 100%)' }}>
          <Heart className="w-10 h-10 text-blue-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-3">Ready to analyse your ECG?</h2>
          <p className="text-blue-300/60 mb-6">Create a free account and upload your first recording.</p>
          <Link to="/register" className="btn-primary">
            Get Started — It's Free <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </div>
  )
}
