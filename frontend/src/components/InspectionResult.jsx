import { CheckCircle2, ClipboardCheck, TriangleAlert, XCircle } from "lucide-react";

import SeverityBadge from "./SeverityBadge";
import { formatReviewStatus, formatSourceType } from "../services/inspectionLabels";

function DecisionIcon({ decision }) {
  if (decision === "Pass") return <CheckCircle2 className="good-icon" size={22} />;
  if (decision === "Fail") return <XCircle className="fail-icon" size={22} />;
  return <TriangleAlert className="review-icon" size={22} />;
}

export default function InspectionResult({ result }) {
  if (!result) {
    return (
      <section className="empty-panel">
        <ClipboardCheck size={28} />
        <p>No inspection selected.</p>
      </section>
    );
  }

  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>Inspection Result</h2>
          <p>{result.model_version || "model pending"}</p>
        </div>
        <DecisionIcon decision={result.pass_fail} />
      </div>

      <div className="result-grid">
        <div className="metric-box">
          <small>Decision</small>
          <strong>{result.pass_fail || "Pending"}</strong>
        </div>
        <div className="metric-box">
          <small>Prediction</small>
          <strong>{result.prediction || "Pending"}</strong>
        </div>
        <div className="metric-box">
          <small>Defect Type</small>
          <strong>{result.defect_type || "Unknown"}</strong>
        </div>
        <div className="metric-box">
          <small>Confidence</small>
          <strong>{result.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : "Pending"}</strong>
        </div>
        <div className="metric-box">
          <small>Severity</small>
          <strong>{result.severity_score != null ? result.severity_score : "Pending"}</strong>
          <SeverityBadge level={result.severity_level} />
        </div>
        <div className="metric-box">
          <small>Anomaly Score</small>
          <strong>{result.anomaly_score != null ? result.anomaly_score.toFixed(2) : "Pending"}</strong>
        </div>
      </div>

      <div className="recommendation">
        <small>Recommended action</small>
        <p>{result.recommended_action || "Waiting for inspection output."}</p>
      </div>

      <div className="metadata-summary">
        <span>
          <strong>Product:</strong> {result.product_id || "Unassigned"}
        </span>
        <span>
          <strong>Batch:</strong> {result.batch_number || "Unassigned"}
        </span>
        <span>
          <strong>Line:</strong> {result.production_line || "Unassigned"}
        </span>
        <span>
          <strong>Shift:</strong> {result.shift || "Unassigned"}
        </span>
        <span>
          <strong>Source:</strong> {result.source_label || formatSourceType(result.source_type)}
        </span>
        <span>
          <strong>Review:</strong> {formatReviewStatus(result.review_status)}
        </span>
      </div>

      <div className="explainability-box">
        <small>AI explainability</small>
        <div className="explainability-grid">
          <span>
            <strong>Threshold:</strong> {result.explainability?.decision_threshold ?? "Pending"}
          </span>
          <span>
            <strong>Defect area:</strong>{" "}
            {result.explainability?.defect_area_percent != null
              ? `${result.explainability.defect_area_percent}%`
              : "Pending"}
          </span>
          <span>
            <strong>Heatmap P95:</strong> {result.explainability?.heatmap_intensity_p95 ?? "Pending"}
          </span>
          <span>
            <strong>Critical zone:</strong> {result.explainability?.critical_location ? "Yes" : "No"}
          </span>
        </div>
        <ul>
          {(result.explainability?.notes || ["No explainability notes recorded."]).map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
