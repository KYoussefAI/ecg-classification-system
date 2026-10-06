import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { historyAPI, errorMessage } from "../utils/api";
import { useAuth } from "../hooks/useAuth";

export default function HistoryPage() {
  const { isAuth } = useAuth();
  const [records, setRecords] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!isAuth) {
      setRecords([]);
      setLoading(false);
      return;
    }
    let active = true;
    historyAPI
      .getAll()
      .then(({ data }) => {
        if (active) setRecords(data);
      })
      .catch((err) => {
        if (active) setError(errorMessage(err));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [isAuth]);
  async function remove(id) {
    try {
      await historyAPI.remove(id);
      setRecords((old) => old.filter((r) => r.id !== id));
    } catch (err) {
      setError(errorMessage(err));
    }
  }
  return (
    <div className="page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">PERSONAL RESEARCH / 03</div>
          <h1>
            Saved Cases<span className="heading-dot">.</span>
          </h1>
          <p>
            Only results you explicitly chose to save. Raw signals are never
            stored.
          </p>
        </div>
      </div>
      {error && (
        <div role="alert" className="error-notice">
          {error}
        </div>
      )}
      {!isAuth ? (
        <section className="panel awaiting">
          <h2>A private space for your research.</h2>
          <p>Guest analyses are not persisted.</p>
          <Link className="btn-primary" to="/login">
            Sign in to view saved cases
          </Link>
        </section>
      ) : loading ? (
        <p role="status">Loading saved cases…</p>
      ) : !records.length ? (
        <section className="panel awaiting">
          <h2>No saved cases yet.</h2>
          <p>Use the save checkbox before running a de-identified analysis.</p>
        </section>
      ) : (
        <section className="panel table-scroll">
          <table>
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Model version</th>
                <th>Above threshold</th>
                <th>Saved (UTC)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.id}>
                  <td>{r.case_id || "Unlabeled case"}</td>
                  <td>{r.model_version}</td>
                  <td>{r.positive_classes.join(", ") || "None"}</td>
                  <td>{r.created_at}</td>
                  <td>
                    <button
                      className="text-button"
                      aria-label={`Delete case ${r.case_id || r.id}`}
                      onClick={() => remove(r.id)}
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
