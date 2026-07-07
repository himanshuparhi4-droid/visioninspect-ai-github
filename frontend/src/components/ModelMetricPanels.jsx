"use client";

import { Cpu, Save, ScanSearch } from "lucide-react";

export function formatMetric(value) {
  if (value == null) return "Pending";
  if (typeof value === "number") return value <= 1 ? value.toFixed(4) : String(value);
  return value;
}

export function ArtifactsPanel({ artifacts = {} }) {
  function artifactLabel(name) {
    if (name === "padim_checkpoint") return "PaDiM checkpoint";
    if (name === "defect_classifier") return "Defect classifier";
    if (name === "baseline_reference") return "Baseline reference";
    return name.replaceAll("_", " ");
  }

  function artifactStatus(name, available) {
    if (available) return { className: "good-text", label: "Available" };
    if (name === "padim_checkpoint") return { className: "severity-pending", label: "External" };
    return { className: "severity-pending", label: "Not loaded" };
  }

  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>Artifacts</h2>
          <p>Backend model availability.</p>
        </div>
        <Cpu size={22} />
      </div>
      <div className="artifact-list">
        {Object.entries(artifacts).map(([name, available]) => {
          const status = artifactStatus(name, available);
          return (
            <div key={name} className="artifact-row">
              <span>{artifactLabel(name)}</span>
              <strong className={status.className}>{status.label}</strong>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function ThresholdSettingsPanel({ settings, message, onChange, onSave }) {
  return (
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
          <ThresholdInput
            label="PaDiM score threshold"
            field="padim_score_threshold"
            step="0.01"
            min="0"
            max="1"
            settings={settings}
            onChange={onChange}
          />
          <ThresholdInput
            label="OpenCV baseline threshold"
            field="baseline_threshold"
            step="0.1"
            min="0"
            max="255"
            settings={settings}
            onChange={onChange}
          />
          <ThresholdInput
            label="Review severity threshold"
            field="review_severity_threshold"
            step="1"
            min="0"
            max="100"
            settings={settings}
            onChange={onChange}
          />
          <ThresholdInput
            label="Fail severity threshold"
            field="fail_severity_threshold"
            step="1"
            min="0"
            max="100"
            settings={settings}
            onChange={onChange}
          />
          <button className="primary-button" type="button" onClick={onSave}>
            <Save size={16} />
            Save thresholds
          </button>
          {message ? (
            <span className={message.includes("Could") ? "inline-error" : "inline-success"}>{message}</span>
          ) : null}
        </div>
      ) : (
        <div className="empty-visual">Loading threshold settings.</div>
      )}
    </section>
  );
}

function ThresholdInput({ label, field, settings, onChange, ...inputProps }) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={settings[field]}
        onChange={(event) => onChange(field, event.target.value)}
        {...inputProps}
      />
    </label>
  );
}

export function ModelComparisonPanel({ models = [] }) {
  return (
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
            {models.map((model) => (
              <tr key={model.name}>
                <td>{model.name}</td>
                <td>{model.task}</td>
                <td>{model.framework}</td>
                <td>
                  {model.primary_metric}: {formatMetric(model.score)}
                </td>
                <td>
                  {model.secondary_metric}: {formatMetric(model.secondary_score)}
                </td>
                <td>{model.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function ThresholdCalibrationPanel({ calibration = {} }) {
  const weakestClass = calibration.weakest_class;

  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>Threshold Calibration</h2>
          <p>Validation metrics and threshold guidance used for decision control.</p>
        </div>
      </div>
      <div className="result-grid">
        <MetricBox label="Evaluation Images" value={calibration.eval_size || "Pending"} />
        <MetricBox label="Accuracy" value={formatMetric(calibration.accuracy)} />
        <MetricBox label="Macro F1" value={formatMetric(calibration.macro_f1)} />
        <div className="metric-box">
          <small>Weakest Class</small>
          <strong>{weakestClass?.label || "Pending"}</strong>
          <small>
            {weakestClass
              ? `F1 ${formatMetric(weakestClass.f1_score)} on ${weakestClass.support} samples`
              : "No classifier report"}
          </small>
        </div>
      </div>
      <div className="explainability-box">
        <strong>Calibration Notes</strong>
        <ul>
          {(calibration.guidance || []).map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function MetricBox({ label, value }) {
  return (
    <div className="metric-box">
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

export function ConfusionMatrixPanel({ labels = [], matrix = [], description = "Saved classifier evaluation." }) {
  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>Confusion Matrix</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Actual \ Predicted</th>
              {labels.map((label) => (
                <th key={label}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, index) => (
              <tr key={labels[index] || index}>
                <td>{labels[index] || index}</td>
                {row.map((value, colIndex) => (
                  <td key={`${index}-${colIndex}`} className={index === colIndex ? "matrix-diagonal" : ""}>
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function ClassifierReportPanel({ report = {} }) {
  const rows = Object.entries(report).filter(([, row]) => typeof row === "object");

  return (
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
            {rows.map(([label, row]) => (
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
  );
}
