"use client";

import { useEffect, useState } from "react";
import { Cpu, Database, Gauge, ShieldCheck, UserCheck } from "lucide-react";

import AppShell from "../../components/AppShell";
import {
  ArtifactsPanel,
  BaselineMetricsTable,
  ClassifierReportPanel,
  ConfusionMatrixPanel,
  ModelComparisonPanel,
  ThresholdCalibrationPanel,
  ThresholdSettingsPanel,
  formatEngine,
  formatPercentMetric,
  metricStatusClass,
} from "../../components/ModelMetricPanels";
import { apiGet } from "../../services/api";
import { getModelMetrics, updateModelSettings } from "../../services/modelApi";

function StatusBadge({ status, note }) {
  return (
    <>
      <strong className={`metric-status ${metricStatusClass(status)}`}>{status || "Unverified"}</strong>
      {note ? <small className="metric-cell-note">{note}</small> : null}
    </>
  );
}

export default function ModelMetricsPage() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [settings, setSettings] = useState(null);
  const [message, setMessage] = useState("");
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    apiGet("/health", { token: null })
      .then(setHealth)
      .catch(() => setHealth(null));
    getModelMetrics()
      .then((payload) => {
        setMetrics(payload);
        setSettings(payload.runtime_settings);
        setLoadError("");
      })
      .catch((error) => {
        setMetrics(null);
        setLoadError(error.message || "Model metrics could not be loaded.");
      });
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

  function updateSetting(field, value) {
    setSettings((current) => ({ ...current, [field]: value }));
  }

  return (
    <AppShell title="Model Metrics" subtitle="Current defect detection and classification model status.">
      {loadError ? (
        <section className="error-panel" role="alert">
          <strong>Model metrics unavailable</strong>
          <p>{loadError}</p>
        </section>
      ) : null}
      <section className="stats-grid">
        <SummaryCard
          icon={ShieldCheck}
          label="Production"
          value={metrics?.release_summary?.production}
          detail="Meets release targets"
        />
        <SummaryCard
          icon={UserCheck}
          label="Manual review"
          value={metrics?.release_summary?.manual_review}
          detail="Subtype below target"
        />
        <SummaryCard
          icon={Gauge}
          label="Binary target"
          value={metrics?.release_summary?.binary_target_met}
          detail="F1 at or above 90%"
        />
        <SummaryCard
          icon={Cpu}
          label="OpenVINO FP16"
          value={metrics?.release_summary?.openvino}
          detail="Active Render engines"
        />
      </section>

      <div className="inspection-layout">
        <ArtifactsPanel artifacts={health?.artifacts || {}} />
        <ThresholdSettingsPanel settings={settings} message={message} onChange={updateSetting} onSave={saveSettings} />
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Category Model Registry</h2>
          <p>Binary detection and defect subtype classification are evaluated and reported separately.</p>
          <p>
            Production requires both stages to pass: Good/Defective detection F1 at or above 90%, and subtype macro F1
            at or above 85%.
          </p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Final status</th>
                <th>Active detector</th>
                <th>Input</th>
                <th>Detection status</th>
                <th>Detection F1</th>
                <th>Detection accuracy</th>
                <th>Defect recall</th>
                <th>Good specificity</th>
                <th>Binary AUROC</th>
                <th>Subtype classifier</th>
                <th>Subtype status</th>
                <th>Subtype accuracy</th>
                <th>Subtype macro F1</th>
                <th>Subtype review threshold</th>
                <th>Runtime size</th>
              </tr>
            </thead>
            <tbody>
              {(metrics?.category_models || []).map((model) => (
                <tr key={model.category}>
                  <td>{model.category.replaceAll("_", " ")}</td>
                  <td>
                    <StatusBadge status={model.release_status} note={model.release_reason} />
                  </td>
                  <td>
                    {formatEngine(model.active_engine)}
                    <small className="metric-cell-note">
                      {model.deployment_precision || "FP32"} · {model.model_version || "v1"}
                    </small>
                    {model.openvino_deferred_for_memory ? (
                      <small className="metric-cell-note">Portable selected for Render memory safety</small>
                    ) : null}
                  </td>
                  <td>{model.input_size ? `${model.input_size} × ${model.input_size}` : "Pending"}</td>
                  <td>
                    <StatusBadge status={model.detection_status} />
                  </td>
                  <td>{formatPercentMetric(model.binary_f1)}</td>
                  <td>{formatPercentMetric(model.binary_accuracy)}</td>
                  <td>{formatPercentMetric(model.binary_recall)}</td>
                  <td>{formatPercentMetric(model.binary_specificity)}</td>
                  <td>{formatPercentMetric(model.binary_auroc)}</td>
                  <td>{formatEngine(model.classifier_engine)}</td>
                  <td>
                    <StatusBadge status={model.subtype_status} />
                  </td>
                  <td>{formatPercentMetric(model.subtype_accuracy)}</td>
                  <td>
                    {formatPercentMetric(model.subtype_macro_f1)}
                    <small className="metric-cell-note">
                      {model.subtype_metric_source}
                      {model.subtype_validation_samples ? ` · ${model.subtype_validation_samples} samples` : ""}
                    </small>
                  </td>
                  <td>{formatPercentMetric(model.subtype_confidence_threshold)}</td>
                  <td>{model.model_size_mb != null ? `${model.model_size_mb.toFixed(1)} MB` : "Pending"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="explainability-box">
          <strong>Why some 90%+ rows still need review</strong>
          <p>
            Detection metrics answer “is this product good or defective?” Subtype metrics answer “which exact defect is
            it?” A category can have strong detection accuracy but still require operator review when subtype macro F1 is
            uneven across smaller defect classes.
          </p>
        </div>
      </section>

      <ModelComparisonPanel models={metrics?.model_comparison || []} />
      <ThresholdCalibrationPanel calibration={metrics?.threshold_calibration || {}} />
      <BaselineMetricsTable rows={metrics?.baseline_metrics || []} />

      <div className="inspection-layout">
        <ConfusionMatrixPanel
          labels={metrics?.confusion_matrix?.labels || []}
          matrix={metrics?.confusion_matrix?.matrix || []}
          description={metrics?.confusion_matrix?.description}
        />
        <ClassifierReportPanel report={metrics?.classifier_report || {}} />
      </div>

      <section className="tool-panel runtime-summary-panel">
        <div className="panel-heading">
          <div>
            <h2>Runtime</h2>
            <p>Live backend, database, storage, and inference readiness.</p>
          </div>
          <Database size={22} />
        </div>
        <div className="result-grid">
          <RuntimeItem label="API" value={health?.status === "ok" ? "Online" : "Unavailable"} />
          <RuntimeItem label="Database" value={health?.database_ready ? "Connected" : "Unavailable"} />
          <RuntimeItem label="Storage" value={health?.storage?.backend || "Unknown"} />
          <RuntimeItem label="Inference" value={health?.inference?.active_engine || "Unknown"} />
        </div>
      </section>
    </AppShell>
  );
}

function SummaryCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="stat-card">
      <span className="stat-icon">
        <Icon size={18} />
      </span>
      <small>{label}</small>
      <strong>{value ?? "-"}</strong>
      <small>{detail}</small>
    </div>
  );
}

function RuntimeItem({ label, value }) {
  return (
    <div className="metric-box">
      <small>{label}</small>
      <strong>{String(value).replaceAll("_", " ")}</strong>
    </div>
  );
}
