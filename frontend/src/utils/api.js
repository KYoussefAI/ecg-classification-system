import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "",
  timeout: 60000,
});
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});
export function errorMessage(error) {
  const detail = error.response?.data?.detail;
  return typeof detail === "string"
    ? detail
    : Array.isArray(detail)
      ? detail.map((x) => x.msg).join(" · ")
      : "Could not reach the API. Check that the backend is running.";
}
export const authAPI = {
  register: (data) => api.post("/api/auth/register", data),
  login: (data) => api.post("/api/auth/login", data),
};
export const predictAPI = {
  predict: (data) => api.post("/api/predict/", data),
  validate: (file) => {
    const body = new FormData();
    body.append("file", file);
    return api.post("/api/predict/validate-file", body);
  },
};
export const historyAPI = {
  getAll: () => api.get("/api/history/"),
  remove: (id) => api.delete(`/api/history/${id}`),
};
export default api;
