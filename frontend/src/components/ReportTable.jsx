import { ExternalLink, FileText } from "lucide-react";

import { downloadReport } from "../services/reportApi";
import { formatDateTime } from "../services/dateTime";

function reportCode(report) {
  const date = report.created_at ? new Date(report.created_at) : null;
  const datePart =
    date && !Number.isNaN(date.getTime())
      ? `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, "0")}${String(date.getDate()).padStart(2, "0")}`
      : "REPORT";
  const suffix = (report.id || report.inspection_id || "0000").slice(-4).toUpperCase();
  return `RPT-${datePart}-${suffix}`;
}

function shortInspectionId(value = "") {
  return value ? `Inspection ${value.slice(0, 8).toUpperCase()}` : "Inspection pending";
}

export default function ReportTable({ reports, onError }) {
  if (!reports?.length) {
    return (
      <section className="empty-panel">
        <FileText size={28} />
        <p>No reports generated yet.</p>
      </section>
    );
  }

  return (
    <div className="report-card-grid">
      {reports.map((report) => (
        <article key={report.id} className="report-card">
          <span className="report-icon">
            <FileText size={20} />
          </span>
          <div>
            <strong>{reportCode(report)}</strong>
            <small>{shortInspectionId(report.inspection_id)}</small>
            <small>{formatDateTime(report.created_at)}</small>
          </div>
          <button
            className="icon-link"
            type="button"
            onClick={() => downloadReport(report).catch((err) => onError?.(err.message || "Could not open report"))}
            aria-label="Open report PDF"
          >
            <ExternalLink size={16} />
          </button>
        </article>
      ))}
    </div>
  );
}
