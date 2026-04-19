import { createContext, useContext, useState, useCallback } from 'react'
import { authAPI } from '../utils/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user,  setUser]  = useState(() => {
    try { return JSON.parse(localStorage.getItem('user')) } catch { return null }
  })
  const [token, setToken] = useState(() => localStorage.getItem('token') || null)

  const login = useCallback(async (username, password) => {
    const res = await authAPI.login({ username, password })
    const { access_token, ...userInfo } = res.data
    localStorage.setItem('token', access_token)
    localStorage.setItem('user',  JSON.stringify(userInfo))
    setToken(access_token)
    setUser(userInfo)
    return userInfo
  }, [])

  const register = useCallback(async (username, email, password) => {
    const res = await authAPI.register({ username, email, password })
    const { access_token, ...userInfo } = res.data
    localStorage.setItem('token', access_token)
    localStorage.setItem('user',  JSON.stringify(userInfo))
    setToken(access_token)
    setUser(userInfo)
    return userInfo
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout, isAuth: !!token }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
