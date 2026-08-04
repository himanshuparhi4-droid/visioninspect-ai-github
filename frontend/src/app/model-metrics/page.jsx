"use client";

import { useEffect, useState } from "react";
import { Database, Gauge } from "lucide-react";

import AppShell from "../../components/AppShell";
import {
  ArtifactsPanel,
  BaselineMetricsTable,
  ClassifierReportPanel,
  ConfusionMatrixPanel,
  ModelComparisonPanel,
  ThresholdCalibrationPanel,
  ThresholdSettingsPanel,
  formatMetric,
} from "../../components/ModelMetricPanels";
import { apiGet } from "../../services/api";
import { getModelMetrics, updateModelSettings } from "../../services/modelApi";

export default function ModelMetricsPage() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [settings, setSettings] = useState(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    apiGet("/health", { token: null })
      .then(setHealth)
      .catch(() => setHealth(null));
    getModelMetrics()
      .then((payload) => {
        setMetrics(payload);
        setSettings(payload.runtime_settings);
      })
      .catch(() => setMetrics(null));
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
        <ArtifactsPanel artifacts={health?.artifacts || {}} />
        <ThresholdSettingsPanel settings={settings} message={message} onChange={updateSetting} onSave={saveSettings} />
      </div>

      <ModelComparisonPanel models={metrics?.model_comparison || []} />
      <ThresholdCalibrationPanel calibration={metrics?.threshold_calibration || {}} />
      <BaselineMetricsTable rows={metrics?.baseline_metrics || []} />

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Category Model Registry</h2>
            <p>Portable and advanced model evidence for every supported product category.</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Runtime</th>
                <th>Type classifier</th>
                <th>Advanced threshold</th>
                <th>Image AUROC</th>
                <th>Live detector F1</th>
                <th>Pixel AUROC</th>
                <th>Classifier accuracy</th>
                <th>Classifier macro F1</th>
              </tr>
            </thead>
            <tbody>
              {(metrics?.category_models || []).map((model) => (
                <tr key={model.category}>
                  <td>{model.category.replaceAll("_", " ")}</td>
                  <td>
                    {model.active_engine
                      ? model.active_engine.replaceAll("_", " ")
                      : model.available
                        ? "Portable baseline"
                        : "Unavailable"}
                  </td>
                  <td>{model.classification_trained ? "Ready" : "Pending"}</td>
                  <td>{formatMetric(model.advanced_decision_threshold)}</td>
                  <td>{formatMetric(model.image_auroc)}</td>
                  <td>{formatMetric(model.openvino_f1 ?? model.portable_cv_f1 ?? model.image_f1)}</td>
                  <td>{formatMetric(model.pixel_auroc)}</td>
                  <td>{formatMetric(model.classifier_accuracy)}</td>
                  <td>{formatMetric(model.classifier_macro_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="inspection-layout">
        <ConfusionMatrixPanel
          labels={metrics?.confusion_matrix?.labels || []}
          matrix={metrics?.confusion_matrix?.matrix || []}
          description={metrics?.confusion_matrix?.description}
        />
        <ClassifierReportPanel report={metrics?.classifier_report || {}} />
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
