import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import {
  Activity, LayoutDashboard, History, LogOut,
  Heart, ChevronRight, User
} from 'lucide-react'

const NAV = [
  { to: '/app/predict',   icon: Activity,         label: 'ECG Analysis'  },
  { to: '/app/dashboard', icon: LayoutDashboard,  label: 'Dashboard'     },
  { to: '/app/history',   icon: History,           label: 'History'       },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => { logout(); navigate('/') }

  return (
    <div className="flex h-screen overflow-hidden ecg-grid">

      {/* ── Sidebar ── */}
      <aside className="w-64 flex-shrink-0 flex flex-col border-r border-blue-900/30 bg-[#080e1c]">
        {/* Logo */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-blue-900/30">
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg"
               style={{ boxShadow: '0 0 20px rgba(59,130,246,0.4)' }}>
            <Heart className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="font-bold text-white leading-none">CardioScan</p>
            <p className="text-[10px] text-blue-400 font-mono mt-0.5">AI · ECG · Analysis</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <p className="text-[10px] uppercase tracking-widest text-blue-600/60 font-semibold px-3 mb-3">
            Navigation
          </p>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span>{label}</span>
              <ChevronRight className="w-3 h-3 ml-auto opacity-40" />
            </NavLink>
          ))}
        </nav>

        {/* User info */}
        <div className="px-4 py-4 border-t border-blue-900/30">
          <div className="flex items-center gap-3 px-2 py-2 rounded-xl bg-blue-900/20 mb-2">
            <div className="w-8 h-8 rounded-lg bg-blue-700/40 flex items-center justify-center">
              <User className="w-4 h-4 text-blue-300" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-white truncate">{user?.username}</p>
              <p className="text-xs text-blue-400/70 truncate">{user?.email}</p>
            </div>
          </div>
          <button onClick={handleLogout} className="btn-danger w-full text-xs py-2">
            <LogOut className="w-3.5 h-3.5" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
