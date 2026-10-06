export default function ResultCard({ result }) {
  if (!result) return null;
  return (
    <section className="panel findings" aria-live="polite">
      <div className="section-heading">
        <div>
          <div className="eyebrow">RESEARCH CLASSIFICATION</div>
          <h2>AI Screening Findings</h2>
        </div>
        <span className="tag">{result.model_version}</span>
      </div>
      <p className="muted">
        Independent model scores. A positive status means the
        validation-selected threshold was crossed.
      </p>
      <div className="finding-grid">
        {result.predictions.map((p) => (
          <article
            className={`finding ${p.positive ? "positive" : ""}`}
            key={p.class}
          >
            <div className="finding-title">
              <span className="target-code">{p.class}</span>
              <span className="tag">
                {p.positive ? "Positive" : "Negative"}
              </span>
            </div>
            <h3>{p.description}</h3>
            <div className="score-number">
              {p.score.toFixed(3)}
              <small>model score</small>
            </div>
            <div className="score-bar">
              <span style={{ width: `${p.score * 100}%` }} />
              <i style={{ left: `${p.threshold * 100}%` }} />
            </div>
            <div className="score-label">
              <span>0</span>
              <span>Threshold {p.threshold.toFixed(3)}</span>
              <span>1</span>
            </div>
          </article>
        ))}
      </div>
      <div className="result-notes">
        {result.warnings.map((w) => (
          <p key={w}>{w}</p>
        ))}
      </div>
      <div className="panel-footer">
        <span>PTB-XL · Train-derived per-lead normalization</span>
        <span>
          {result.id ? "Saved to your private case history" : "Not saved"}
        </span>
      </div>
    </section>
  );
}
