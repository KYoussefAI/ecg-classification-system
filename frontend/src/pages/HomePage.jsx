import { Link } from "react-router-dom";
import { Activity, ArrowUpRight, ArrowRight } from "lucide-react";
import { useMemo } from "react";
import Waveform from "../components/Waveform";
import Disclaimer from "../components/Disclaimer";
import { syntheticSignal } from "../utils/synthetic";

const TARGETS = [
  ["CD", "Conduction disturbance", "Electrical conduction patterns"],
  ["HYP", "Hypertrophy", "Patterns associated with enlargement"],
  ["MI", "Myocardial infarction", "Infarction-associated ECG patterns"],
  ["NORM", "Normal ECG pattern", "Dataset-defined normal superclass"],
  ["STTC", "ST/T change", "Repolarization-associated patterns"],
];

export default function HomePage() {
  const signal = useMemo(syntheticSignal, []);
  return (
    <div className="landing">
      <header className="landing-nav">
        <Link to="/" className="brand">
          <Activity />
          <span>
            CardioScan<small>SIGNAL INTELLIGENCE</small>
          </span>
        </Link>
        <nav>
          <a href="#methodology">Methodology</a>
          <Link to="/app/dashboard">Model performance</Link>
          <Link className="nav-enter" to="/app/predict">
            Enter workspace <ArrowUpRight size={16} />
          </Link>
        </nav>
      </header>
      <section className="hero">
        <div className="eyebrow">
          <span className="live-dot" /> OPEN RESEARCH · 12-LEAD ECG
        </div>
        <h1>
          Every lead.
          <br />
          <span>A clearer perspective.</span>
        </h1>
        <p className="hero-copy">
          An intelligent workspace for the cardiac signal.
          <br />
          AI-assisted 12-lead ECG classification research prototype.
        </p>
        <div className="hero-actions">
          <Link className="btn-primary" to="/app/predict">
            Open ECG Workstation <ArrowRight size={17} />
          </Link>
          <a className="text-button" href="#methodology">
            Explore the methodology <ArrowUpRight size={16} />
          </a>
        </div>
        <div className="hero-monitor">
          <div className="monitor-header">
            <span>
              <i className="live-dot" /> SIGNAL PREVIEW
            </span>
            <span>SYNTHETIC DEMO SIGNAL · LEAD II</span>
          </div>
          <Waveform signal={signal} compact />
          <div className="monitor-bottom">
            <span>
              <b>12</b> standard leads
            </span>
            <span>
              <b>100</b> Hz input
            </span>
            <span>
              <b>10</b> seconds
            </span>
            <span className="muted">
              Illustrative signal · not patient data
            </span>
          </div>
        </div>
      </section>
      <section className="method-strip" aria-label="Pipeline">
        {[
          "PTB-XL",
          "ResNet1D",
          "Multi-label classification",
          "Held-out evaluation",
        ].map((x, i) => (
          <div key={x}>
            <small>0{i + 1}</small>
            {x}
            {i < 3 && <ArrowRight size={16} />}
          </div>
        ))}
      </section>
      <section className="landing-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">FIVE DIAGNOSTIC SUPERCLASSES</div>
            <h2>One signal. Multiple patterns.</h2>
          </div>
          <p>
            Independent classifier outputs.
            <br />
            Interpreted transparently, without diagnostic claims.
          </p>
        </div>
        <div className="target-grid">
          {TARGETS.map(([code, name, desc]) => (
            <article key={code}>
              <span className="target-code">{code}</span>
              <h3>{name}</h3>
              <p>{desc}</p>
            </article>
          ))}
        </div>
      </section>
      <section className="landing-section transparency" id="methodology">
        <div>
          <div className="eyebrow">RESEARCH, IN THE OPEN</div>
          <h2>
            Built to be
            <br />
            examined.
          </h2>
          <p>
            From patient-separated folds to visible decision thresholds, the
            experiment is part of the interface.
          </p>
          <Link className="text-button" to="/app/dashboard">
            View model provenance <ArrowUpRight size={16} />
          </Link>
        </div>
        <div className="method-list">
          {[
            [
              "01",
              "Dataset",
              "PTB-XL clinical recordings. Exact included counts are generated with each run.",
            ],
            [
              "02",
              "Patient-separated split",
              "Folds 1–8 train. Fold 9 selects the checkpoint and thresholds. Fold 10 is reserved for final evaluation.",
            ],
            [
              "03",
              "Temporal residual network",
              "A wider receptive field, progressive downsampling, and five independent output scores.",
            ],
            [
              "04",
              "Honest evaluation",
              "Metrics appear only after the active model has been evaluated. No historical numbers are substituted.",
            ],
            [
              "05",
              "Known limitations",
              "Uncalibrated scores, potential domain shift, and multilabel inconsistencies. No clinical validation.",
            ],
          ].map(([number, title, body]) => (
            <article key={number}>
              <span>{number}</span>
              <div>
                <h3>{title}</h3>
                <p>{body}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
      <Disclaimer />
    </div>
  );
}
