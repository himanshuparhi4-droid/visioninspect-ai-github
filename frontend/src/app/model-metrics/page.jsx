"use client";

import { useEffect, useState } from "react";
import { Cpu, Database, Gauge, Save, ScanSearch } from "lucide-react";

import AppShell from "../../components/AppShell";
import { apiGet } from "../../services/api";
import { getModelMetrics, updateModelSettings } from "../../services/modelApi";

function formatMetric(value) {
  if (value == null) return "Pending";
  if (typeof value === "number") return value <= 1 ? value.toFixed(4) : String(value);
  return value;
}

export default function ModelMetricsPage() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [settings, setSettings] = useState(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    apiGet("/health", { token: null }).then(setHealth).catch(() => setHealth(null));
    getModelMetrics().then((payload) => {
      setMetrics(payload);
      setSettings(payload.runtime_settings);
    }).catch(() => setMetrics(null));
  }, []);

  async function saveSettings() {
    setMessage("");
    try {
      const saved = await updateModelSettings({
        padim_score_threshold: Number(settings.padim_score_threshold),
        baseline_threshold: Number(settings.baseline_threshold),
        review_severity_threshold: Number(settings.review_severity_threshold),
        fail_severity_threshold: Number(settings.fail_severity_threshold),
      });
      setSettings(saved);
      setMessage("Threshold settings saved");
    } catch (err) {
      setMessage(err.message || "Could not save settings");
    }
  }

  const classifierLabels = metrics?.confusion_matrix?.labels || [];
  const confusionMatrix = metrics?.confusion_matrix?.matrix || [];
  const classifierReport = metrics?.classifier_report || {};

  return (
    <AppShell title="Model Metrics" subtitle="Current defect detection and classification model status.">
      <section className="stats-grid">
        {(metrics?.model_comparison || []).map((model) => (
          <div key={model.name} className="stat-card">
            <span className="stat-icon">
              <Gauge size={18} />
            </span>
            <small>{model.primary_metric}</small>
            <strong>{formatMetric(model.score)}</strong>
            <small>{model.name}</small>
          </div>
        ))}
      </section>

      <div className="inspection-layout">
        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Artifacts</h2>
              <p>Backend model availability.</p>
            </div>
            <Cpu size={22} />
          </div>
          <div className="artifact-list">
            {Object.entries(health?.artifacts || {}).map(([name, available]) => (
              <div key={name} className="artifact-row">
                <span>{name.replaceAll("_", " ")}</span>
                <strong className={available ? "good-text" : "fail-text"}>{available ? "Available" : "Missing"}</strong>
              </div>
            ))}
          </div>
        </section>

        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Threshold Tuning</h2>
              <p>Admin-controlled decision thresholds for live inference.</p>
            </div>
            <Save size={22} />
          </div>
          {settings ? (
            <div className="compact-form">
              <label>
                PaDiM score threshold
                <input type="number" step="0.01" min="0" max="1" value={settings.padim_score_threshold} onChange={(event) => setSettings((current) => ({ ...current, padim_score_threshold: event.target.value }))} />
              </label>
              <label>
                OpenCV baseline threshold
                <input type="number" step="0.1" min="0" max="255" value={settings.baseline_threshold} onChange={(event) => setSettings((current) => ({ ...current, baseline_threshold: event.target.value }))} />
              </label>
              <label>
                Review severity threshold
                <input type="number" step="1" min="0" max="100" value={settings.review_severity_threshold} onChange={(event) => setSettings((current) => ({ ...current, review_severity_threshold: event.target.value }))} />
              </label>
              <label>
                Fail severity threshold
                <input type="number" step="1" min="0" max="100" value={settings.fail_severity_threshold} onChange={(event) => setSettings((current) => ({ ...current, fail_severity_threshold: event.target.value }))} />
              </label>
              <button className="primary-button" type="button" onClick={saveSettings}>
                <Save size={16} />
                Save thresholds
              </button>
              {message ? <span className={message.includes("Could") ? "inline-error" : "inline-success"}>{message}</span> : null}
            </div>
          ) : <div className="empty-visual">Loading threshold settings.</div>}
        </section>
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Model Comparison</h2>
            <p>Detection, classification, and fallback model roles.</p>
          </div>
          <ScanSearch size={22} />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Task</th>
                <th>Framework</th>
                <th>Primary</th>
                <th>Secondary</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {(metrics?.model_comparison || []).map((model) => (
                <tr key={model.name}>
                  <td>{model.name}</td>
                  <td>{model.task}</td>
                  <td>{model.framework}</td>
                  <td>{model.primary_metric}: {formatMetric(model.score)}</td>
                  <td>{model.secondary_metric}: {formatMetric(model.secondary_score)}</td>
                  <td>{model.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="inspection-layout">
        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Confusion Matrix</h2>
              <p>Saved classifier evaluation.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Actual \ Predicted</th>
                  {classifierLabels.map((label) => <th key={label}>{label}</th>)}
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, index) => (
                  <tr key={classifierLabels[index] || index}>
                    <td>{classifierLabels[index] || index}</td>
                    {row.map((value, colIndex) => <td key={`${index}-${colIndex}`}>{value}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Classifier Report</h2>
              <p>Precision, recall, and F1 by class.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(classifierReport).filter(([, row]) => typeof row === "object").map(([label, row]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{formatMetric(row.precision)}</td>
                    <td>{formatMetric(row.recall)}</td>
                    <td>{formatMetric(row["f1-score"])}</td>
                    <td>{row.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Runtime</h2>
            <p>Service health from FastAPI.</p>
          </div>
          <Database size={22} />
        </div>
        <pre className="json-panel">{JSON.stringify({ health, runtime_settings: settings }, null, 2)}</pre>
      </section>
    </AppShell>
  );
}
