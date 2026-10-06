import { useRef, useState } from "react";
import { Activity, Upload, Check, ArrowRight, RotateCcw } from "lucide-react";
import { predictAPI, errorMessage } from "../utils/api";
import { useAuth } from "../hooks/useAuth";
import useModel from "../hooks/useModel";
import { syntheticSignal, LEADS } from "../utils/synthetic";
import Waveform from "../components/Waveform";
import ResultCard from "../components/ResultCard";

export default function PredictPage() {
  const { isAuth } = useAuth();
  const { metadata, error: offline } = useModel();
  const [record, setRecord] = useState(null);
  const [filename, setFilename] = useState("");
  const [caseId, setCaseId] = useState("");
  const [save, setSave] = useState(false);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  async function loadFile(file) {
    if (!file) return;
    setRecord(null);
    setResult(null);
    setError("");
    setBusy(true);
    try {
      if (file.size > 2 * 1024 * 1024)
        throw new Error("File must be smaller than 2 MiB");
      const { data } = await predictAPI.validate(file);
      setRecord(data);
      setFilename(file.name);
    } catch (err) {
      setError(err.response ? errorMessage(err) : err.message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }
  function demo() {
    setRecord({
      signal_data: syntheticSignal(),
      leads: LEADS,
      sample_rate: 100,
      units: "mV",
      warnings: [
        "Synthetic demo signal — illustrative morphology, not a patient recording.",
      ],
      synthetic: true,
    });
    setFilename("Synthetic demo signal");
    setResult(null);
    setError("");
  }
  async function analyze() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const { data } = await predictAPI.predict({
        signal_data: record.signal_data,
        leads: record.leads,
        sample_rate: record.sample_rate,
        units: record.units,
        case_id: caseId || null,
        save_history: isAuth && save,
        synthetic: !!record.synthetic,
      });
      setResult(data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="page workstation">
      <div className="page-heading">
        <div>
          <div className="eyebrow">SIGNAL WORKSPACE / 01</div>
          <h1>
            ECG Workstation<span className="heading-dot">.</span>
          </h1>
          <p>Inspect the signal. Explore the model. Keep the context.</p>
        </div>
        <span className="tag">12-LEAD · MULTI-LABEL</span>
      </div>
      <div className="privacy-note">
        Use de-identified data only. An optional Case ID should never contain a
        patient name or other identifying information.
      </div>
      <section className="panel acquisition">
        <div>
          <span className="step-number">01</span>
          <h2>Load a recording</h2>
          <p>JSON, CSV or NPY · 1000 × 12 · 100 Hz · mV</p>
        </div>
        <div className="acquisition-actions">
          <input
            ref={fileRef}
            type="file"
            accept=".json,.csv,.npy"
            onChange={(e) => loadFile(e.target.files[0])}
            hidden
          />
          <button
            className="btn-secondary"
            disabled={busy}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={16} />
            Upload ECG
          </button>
          <button className="text-button" disabled={busy} onClick={demo}>
            Load synthetic demo <ArrowRight size={16} />
          </button>
        </div>
      </section>
      <p className="format-note">
        Standard order: I, II, III, aVR, aVL, aVF, V1–V6. Files without metadata
        declare 100 Hz, mV and this lead order. Unsupported formats are
        rejected.
      </p>
      {error && (
        <div className="error-notice" role="alert">
          {error}
        </div>
      )}
      {!metadata?.model_loaded && (
        <div className="notice">
          {offline ? "The API is offline." : "No trained model is available."}{" "}
          You can preview the synthetic signal; prediction requires a valid
          trained artifact.
        </div>
      )}
      {record ? (
        <>
          <section className="panel signal-panel">
            <div className="signal-toolbar">
              <div>
                <div className="eyebrow">
                  {record.synthetic
                    ? "SYNTHETIC DEMO SIGNAL"
                    : "LOADED RECORDING"}
                </div>
                <h2>{caseId || filename}</h2>
              </div>
              <div className="signal-specs">
                <span>
                  <b>12</b> leads
                </span>
                <span>
                  <b>{record.sample_rate}</b> Hz
                </span>
                <span>
                  <b>10</b> s
                </span>
                <span className="quality-pass">
                  <Check size={14} /> Basic checks passed
                </span>
              </div>
            </div>
            <Waveform signal={record.signal_data} analyzing={busy} />
            <div className="quality-checklist">
              {[
                "12 leads detected",
                "10 s duration",
                "100 Hz",
                "Finite samples",
                "No flatline leads",
              ].map((c) => (
                <span key={c}>
                  <Check size={13} />
                  {c}
                </span>
              ))}
            </div>
          </section>
          {record.warnings?.map((w) => (
            <p className="format-note" key={w}>
              {w}
            </p>
          ))}
          <section className="panel analysis-controls">
            <div>
              <label className="label" htmlFor="case-id">
                Case ID <span className="muted">(optional)</span>
              </label>
              <input
                id="case-id"
                className="input"
                placeholder="e.g. DEMO-001"
                maxLength={100}
                value={caseId}
                onChange={(e) => setCaseId(e.target.value)}
              />
              <label className="save-option">
                <input
                  type="checkbox"
                  checked={isAuth && save}
                  disabled={!isAuth}
                  onChange={(e) => setSave(e.target.checked)}
                />
                {isAuth
                  ? "Save this de-identified result to my history"
                  : "Sign in to save results; guest analyses are never saved"}
              </label>
            </div>
            <div className="run-controls">
              <small>Model: {metadata?.model_version || "unavailable"}</small>
              <button
                className="btn-primary"
                onClick={analyze}
                disabled={busy || !metadata?.model_loaded}
              >
                <Activity size={17} />
                {busy ? "Analyzing signal…" : "Run research classification"}
              </button>
            </div>
          </section>
          <ResultCard result={result} />
          <button
            className="text-button reset-button"
            disabled={busy}
            onClick={() => {
              setRecord(null);
              setResult(null);
              setCaseId("");
              setSave(false);
            }}
          >
            <RotateCcw size={14} />
            Clear workspace
          </button>
        </>
      ) : (
        <div className="empty-signal">
          <Activity size={42} />
          <h2>Your signal belongs here.</h2>
          <p>
            Load a recording to reveal all twelve leads.
            <br />
            Waveforms are previewed before classification.
          </p>
          <div className="empty-ruler">
            <span>0 s</span>
            <span>5 s</span>
            <span>10 s</span>
          </div>
        </div>
      )}
    </div>
  );
}
