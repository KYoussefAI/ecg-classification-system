import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Heart, Eye, EyeOff, AlertCircle, Loader2, CheckCircle } from 'lucide-react'
import { useAuth } from '../hooks/useAuth'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate     = useNavigate()

  const [form,    setForm]    = useState({ username: '', email: '', password: '', confirm: '' })
  const [showPw,  setShowPw]  = useState(false)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')

  const passwordOk = form.password.length >= 6
  const match      = form.password === form.confirm && form.confirm.length > 0

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!match) { setError('Passwords do not match.'); return }
    setError('')
    setLoading(true)
    try {
      await register(form.username, form.email, form.password)
      navigate('/app/predict')
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen ecg-grid flex items-center justify-center px-4">
      <div className="w-full max-w-md animate-slide-up">

        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-blue-600 flex items-center justify-center mb-4"
               style={{ boxShadow: '0 0 30px rgba(59,130,246,0.5)' }}>
            <Heart className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Create your account</h1>
          <p className="text-blue-400/60 text-sm mt-1">Start analysing ECGs for free</p>
        </div>

        <div className="card card-glow">
          {error && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-900/20 border border-red-700/30 text-red-300 text-sm mb-5">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="label">Username</label>
              <input className="input" placeholder="john_doe" value={form.username}
                onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
                required autoFocus minLength={3} />
            </div>

            <div>
              <label className="label">Email</label>
              <input className="input" type="email" placeholder="john@hospital.com" value={form.email}
                onChange={e => setForm(f => ({ ...f, email: e.target.value }))} required />
            </div>

            <div>
              <label className="label">Password</label>
              <div className="relative">
                <input className="input pr-11" type={showPw ? 'text' : 'password'}
                  placeholder="Min 6 characters" value={form.password}
                  onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                  required minLength={6} />
                <button type="button" onClick={() => setShowPw(p => !p)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-blue-400/50 hover:text-blue-300">
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {form.password.length > 0 && (
                <p className={`text-xs mt-1 ${passwordOk ? 'text-green-400' : 'text-red-400'}`}>
                  {passwordOk ? '✓ Strong enough' : '✗ At least 6 characters'}
                </p>
              )}
            </div>

            <div>
              <label className="label">Confirm Password</label>
              <div className="relative">
                <input className="input pr-11" type={showPw ? 'text' : 'password'}
                  placeholder="Repeat password" value={form.confirm}
                  onChange={e => setForm(f => ({ ...f, confirm: e.target.value }))} required />
                {form.confirm.length > 0 && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2">
                    {match
                      ? <CheckCircle className="w-4 h-4 text-green-400" />
                      : <AlertCircle className="w-4 h-4 text-red-400" />}
                  </span>
                )}
              </div>
            </div>

            <button type="submit" className="btn-primary w-full mt-2" disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {loading ? 'Creating account…' : 'Create Account'}
            </button>
          </form>

          <p className="text-center text-sm text-blue-400/60 mt-5">
            Already have an account?{' '}
            <Link to="/login" className="text-blue-400 hover:text-blue-300 font-medium">Sign in</Link>
          </p>
        </div>

        <p className="text-center mt-4">
          <Link to="/" className="text-xs text-blue-600/50 hover:text-blue-400">← Back to home</Link>
        </p>
      </div>
    </div>
  )
}
