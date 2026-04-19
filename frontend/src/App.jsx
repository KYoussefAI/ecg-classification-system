import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './hooks/useAuth'

import Layout       from './components/Layout'
import HomePage     from './pages/HomePage'
import LoginPage    from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import PredictPage  from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
import HistoryPage  from './pages/HistoryPage'

function PrivateRoute({ children }) {
  const { isAuth } = useAuth()
  return isAuth ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public routes */}
          <Route path="/"         element={<HomePage />} />
          <Route path="/login"    element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected routes inside shared Layout */}
          <Route path="/app" element={<PrivateRoute><Layout /></PrivateRoute>}>
            <Route index          element={<Navigate to="/app/predict" replace />} />
            <Route path="predict"   element={<PredictPage />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="history"   element={<HistoryPage />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
