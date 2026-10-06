import { Link } from "react-router-dom";
import useModel from "../hooks/useModel";

const fmt = (value) => (typeof value === "number" ? value.toFixed(3) : "—");
const COLORS = ["#69d6d0", "#80aaff", "#e1b880", "#c0d6df", "#839ba9"];

function Curve({ title, curves }) {
  return (
    <article className="panel curve">
      <h3>{title}</h3>
      <svg
        viewBox="0 0 360 235"
        role="img"
        aria-label={`${title} curves for the evaluated model`}
      >
        <path d="M35 15 V205 H345" fill="none" stroke="#647683" />
        <path
          d="M35 110 H345 M190 15 V205"
          stroke="#243440"
          strokeDasharray="3 5"
        />
        {Object.entries(curves).map(([cls, points], i) => (
          <polyline
            key={cls}
            points={points
              .map(([x, y]) => `${35 + x * 310},${205 - y * 190}`)
              .join(" ")}
            fill="none"
            stroke={COLORS[i]}
            strokeWidth="1.7"
          />
        ))}
        <text x="32" y="222">
          0
        </text>
        <text x="335" y="222">
          1
        </text>
        <text x="16" y="22">
          1
        </text>
        <text x="130" y="232">
          {title === "ROC" ? "False positive rate" : "Recall"}
        </text>
        <text transform="translate(10 160) rotate(-90)">
          {title === "ROC" ? "Sensitivity" : "Precision"}
        </text>
      </svg>
      <div className="curve-legend">
        {Object.keys(curves).map((cls, i) => (
          <span key={cls} style={{ color: COLORS[i] }}>
            {cls}
          </span>
        ))}
      </div>
    </article>
  );
}

export default function DashboardPage() {
  const { metadata, error } = useModel();
  const metrics = metadata?.test_metrics;
  return (
    <div className="page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">RESEARCH TRANSPARENCY / 02</div>
          <h1>
            Model Performance<span className="heading-dot">.</span>
          </h1>
          <p>Held-out PTB-XL Fold 10</p>
        </div>
        <span className="tag">
          {metadata?.model_version || "NO ACTIVE MODEL"}
        </span>
      </div>
      {error && (
        <div role="alert" className="error-notice">
          The API is offline. Performance cannot be verified.
        </div>
      )}
      {!metrics ? (
        <section className="panel awaiting">
          <span className="eyebrow">EVIDENCE BEFORE NUMBERS</span>
          <h2>Awaiting final evaluation</h2>
          <p>
            Metrics appear here only when the active checkpoint has completed
            its held-out evaluation. No historical performance is shown.
          </p>
          <Link className="text-button" to="/app/predict">
            Open the signal workspace →
          </Link>
        </section>
      ) : (
        <>
          <div className="metric-grid">
            {[
              ["Macro AUROC", fmt(metrics.macro_auroc)],
              ["Macro AUPRC", fmt(metrics.macro_auprc)],
              ["Macro F1", fmt(metrics.macro_f1)],
              ["Test records", metrics.record_count.toLocaleString()],
            ].map(([label, value]) => (
              <article className="panel metric" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
                <small>Held-out fold 10</small>
              </article>
            ))}
          </div>
          {metrics.auroc_ci95?.intervals?.macro && (
            <p className="format-note">
              Macro AUROC 95% patient-bootstrap interval:{" "}
              {fmt(metrics.auroc_ci95.intervals.macro.lower)}–
              {fmt(metrics.auroc_ci95.intervals.macro.upper)}. Conditional on
              this frozen model; does not measure training variability.
            </p>
          )}
          <section className="panel">
            <h2>Per-class evaluation</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {[
                      "Class",
                      "AUROC",
                      "AUPRC",
                      "Precision",
                      "Sensitivity",
                      "Specificity",
                      "F1",
                      "Support",
                      "Threshold",
                    ].map((x) => (
                      <th key={x}>{x}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metadata.classes.map((cls) => {
                    const p = metrics.per_class[cls];
                    return (
                      <tr key={cls}>
                        <td>{cls}</td>
                        {[
                          "auroc",
                          "auprc",
                          "precision",
                          "recall",
                          "specificity",
                          "f1",
                        ].map((k) => (
                          <td key={k}>{fmt(p[k])}</td>
                        ))}
                        <td>{p.support}</td>
                        <td>{fmt(p.threshold)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
          {metadata.curves && (
            <div className="curve-grid">
              <Curve title="ROC" curves={metadata.curves.roc} />
              <Curve
                title="Precision–recall"
                curves={metadata.curves.precision_recall}
              />
            </div>
          )}
        </>
      )}
      <section className="panel provenance">
        <div>
          <div className="eyebrow">MODEL PROVENANCE</div>
          <h2>Every decision has a source.</h2>
        </div>
        <dl>
          <div>
            <dt>Dataset</dt>
            <dd>
              {metadata?.dataset
                ? `${metadata.dataset.name} v${metadata.dataset.version}`
                : "PTB-XL · training pending"}
            </dd>
          </div>
          <div>
            <dt>Architecture</dt>
            <dd>{metadata?.architecture || "ResNet1D · pending artifact"}</dd>
          </div>
          <div>
            <dt>Training / validation / test</dt>
            <dd>Folds 1–8 / 9 / 10</dd>
          </div>
          <div>
            <dt>Decision thresholds</dt>
            <dd>Per-class F1 on validation fold 9 only</dd>
          </div>
          <div>
            <dt>Output semantics</dt>
            <dd>Uncalibrated model scores</dd>
          </div>
        </dl>
        {metadata?.thresholds && (
          <div className="threshold-list">
            {metadata.classes.map((cls) => (
              <span key={cls}>
                {cls} <b>{fmt(metadata.thresholds[cls])}</b>
              </span>
            ))}
          </div>
        )}
      </section>
      <p className="format-note">
        Dataset benchmark performance does not establish clinical effectiveness.
        AUPRC uses average precision. Scores may shift across populations,
        devices and acquisition settings.
      </p>
    </div>
  );
}
