import { CheckCircle2, ClipboardCheck, TriangleAlert, XCircle } from "lucide-react";

import SeverityBadge from "./SeverityBadge";
import { formatConfidence, formatReviewStatus, formatSourceType } from "../services/inspectionLabels";

function DecisionIcon({ decision }) {
  if (decision === "Pass") return <CheckCircle2 className="good-icon" size={22} />;
  if (decision === "Fail") return <XCircle className="fail-icon" size={22} />;
  return <TriangleAlert className="review-icon" size={22} />;
}

function formatScore(value) {
  if (value == null) return "Pending";
  return value < 0.1 ? value.toFixed(4) : value.toFixed(2);
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

  const inferenceMs = result.explainability?.runtime_ms?.inference;

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
          <small>Decision Confidence</small>
          <strong>{formatConfidence(result.confidence)}</strong>
        </div>
        <div className="metric-box">
          <small>Severity</small>
          <strong>{result.severity_score != null ? result.severity_score : "Pending"}</strong>
          <SeverityBadge level={result.severity_level} />
        </div>
        <div className="metric-box">
          <small>Anomaly Score</small>
          <strong>{formatScore(result.anomaly_score)}</strong>
        </div>
      </div>

      <div className="runtime-strip">
        <span>
          <small>Detector</small>
          <strong>{(result.detector_engine || result.explainability?.engine || "Unknown").replaceAll("_", " ")}</strong>
        </span>
        <span>
          <small>Subtype Model</small>
          <strong>
            {(result.classifier_engine || result.explainability?.classifier_engine || "Not applicable").replaceAll(
              "_",
              " "
            )}
          </strong>
        </span>
        <span>
          <small>Model Category</small>
          <strong>{(result.category || "Unknown").replaceAll("_", " ")}</strong>
        </span>
        <span>
          <small>Subtype Assurance</small>
          <strong>
            {result.prediction === "Good" ? "Not applicable" : result.subtype_model_status || "Unverified"}
          </strong>
        </span>
        <span>
          <small>Inference Time</small>
          <strong>{inferenceMs != null ? `${Number(inferenceMs).toFixed(1)} ms` : "Not recorded"}</strong>
        </span>
      </div>

      {result.manual_review_required ? (
        <div className="model-quality-warning" role="status">
          <TriangleAlert size={17} />
          <span>
            <strong>Manual subtype review required</strong>
            <small>
              The Good/Defective decision remains active, but this category&apos;s subtype model is below the release
              target or could not identify the defect reliably.
            </small>
          </span>
        </div>
      ) : null}

      {result.detector_fallback_used || result.classifier_fallback_used ? (
        <div className="model-fallback-warning" role="status">
          <TriangleAlert size={17} />
          <span>
            <strong>Compatibility fallback active</strong>
            <small>
              {result.detector_fallback_reason ||
                result.classifier_fallback_reason ||
                "A preferred model was unavailable."}
            </small>
          </span>
        </div>
      ) : null}

      <div className="recommendation">
        <small>Recommended action</small>
        <p>{result.recommended_action || "Waiting for inspection output."}</p>
      </div>

      <div className="metadata-summary">
        <span>
          <strong>Product:</strong> {result.product_id || "Unassigned"}
        </span>
        <span>
          <strong>Category:</strong> {(result.category || "Unknown").replaceAll("_", " ")}
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
            <strong>Threshold:</strong> {formatScore(result.explainability?.decision_threshold)}
          </span>
          <span>
            <strong>Detection confidence:</strong>{" "}
            {result.explainability?.detection_confidence != null
              ? formatConfidence(result.explainability.detection_confidence)
              : "Pending"}
          </span>
          <span>
            <strong>
              {result.classification_confidence_calibrated ? "Subtype reliability:" : "Subtype confidence:"}
            </strong>{" "}
            {formatConfidence(result.explainability?.classification_confidence, "Not applicable")}
          </span>
          {result.raw_classification_confidence != null && result.classification_confidence_calibrated ? (
            <span>
              <strong>Raw model score:</strong> {`${(result.raw_classification_confidence * 100).toFixed(1)}%`}
            </span>
          ) : null}
          <span>
            <strong>Confidence calibration:</strong>{" "}
            {result.classification_confidence == null
              ? "Not applicable"
              : result.classification_confidence_calibrated
                ? "Calibrated"
                : "Raw probability"}
          </span>
          <span>
            <strong>Subtype model macro F1:</strong>{" "}
            {result.subtype_model_macro_f1 != null
              ? `${(result.subtype_model_macro_f1 * 100).toFixed(1)}%`
              : "Unverified"}
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
