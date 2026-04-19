import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Handle 401 — clear token and redirect
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// ── Auth ──────────────────────────────────────────────────────────────────────
export const authAPI = {
  register: (data) => api.post('/api/auth/register', data),
  login:    (data) => api.post('/api/auth/login',    data),
}

// ── Prediction ────────────────────────────────────────────────────────────────
export const predictAPI = {
  predict: (data) => api.post('/api/predict/', data),
}

// ── History ───────────────────────────────────────────────────────────────────
export const historyAPI = {
  getAll:  (limit = 50)  => api.get(`/api/history/?limit=${limit}`),
  remove:  (id)          => api.delete(`/api/history/${id}`),
}

// ── Stats ─────────────────────────────────────────────────────────────────────
export const statsAPI = {
  get: () => api.get('/api/stats/'),
}

export default api
