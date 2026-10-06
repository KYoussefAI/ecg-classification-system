import { Link, NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  BarChart2,
  History,
  LogOut,
  ArrowUpRight,
} from "lucide-react";
import { useAuth } from "../hooks/useAuth";
import useModel from "../hooks/useModel";
import Disclaimer from "./Disclaimer";

export default function Layout() {
  const { user, isAuth, logout } = useAuth();
  const { metadata, error } = useModel();
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" to="/">
          <Activity size={25} />
          <span>
            CardioScan<small>SIGNAL INTELLIGENCE</small>
          </span>
        </Link>
        <div className="sidebar-label">RESEARCH WORKSPACE</div>
        <nav aria-label="Workspace navigation">
          <NavLink to="/app/predict">
            <Activity size={18} />
            ECG Workstation
          </NavLink>
          <NavLink to="/app/dashboard">
            <BarChart2 size={18} />
            Model Performance
          </NavLink>
          <NavLink to="/app/history">
            <History size={18} />
            Saved Cases
          </NavLink>
        </nav>
        <div className="sidebar-note">
          <span className="tiny-cross">+</span>
          <p>
            Signals first.
            <br />
            Transparent by design.
          </p>
          <small>PTB-XL · 12-lead research</small>
        </div>
        <div className="account">
          <span>{isAuth ? user?.username : "Guest workspace"}</span>
          <small>
            {isAuth
              ? "Save only de-identified cases"
              : "Analyses are not saved"}
          </small>
          {isAuth ? (
            <button className="text-button" onClick={logout}>
              <LogOut size={14} />
              Sign out
            </button>
          ) : (
            <Link to="/login">
              Sign in <ArrowUpRight size={14} />
            </Link>
          )}
        </div>
      </aside>
      <div className="main-shell">
        <header className="workspace-topbar">
          <span>
            ECG LAB <span className="muted">/ Research classification</span>
          </span>
          <span className={`status ${metadata?.model_loaded ? "ready" : ""}`}>
            <i />
            {error
              ? "API offline"
              : metadata?.model_loaded
                ? "Model ready"
                : "Model unavailable"}
          </span>
        </header>
        <main>
          <Outlet />
        </main>
        <Disclaimer />
      </div>
    </div>
  );
}
