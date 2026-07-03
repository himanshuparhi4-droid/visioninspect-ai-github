import { ExternalLink, FileText } from "lucide-react";

import { downloadReport } from "../services/reportApi";
import { formatDateTime } from "../services/dateTime";

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
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Report</th>
            <th>Inspection</th>
            <th>Type</th>
            <th>Created</th>
            <th>Open</th>
          </tr>
        </thead>
        <tbody>
          {reports.map((report) => (
            <tr key={report.id}>
              <td>{report.id}</td>
              <td>{report.inspection_id}</td>
              <td>{report.report_type}</td>
              <td>{formatDateTime(report.created_at)}</td>
              <td>
                <button
                  className="icon-link"
                  type="button"
                  onClick={() => downloadReport(report).catch((err) => onError?.(err.message || "Could not open report"))}
                  aria-label="Open report PDF"
                >
                  <ExternalLink size={16} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
