import { apiDownloadBlob, apiGet, apiPost } from "./api";

export function createInspectionReport(inspectionId) {
  return apiPost(`/reports/inspection/${inspectionId}`, {});
}

export function listReports() {
  return apiGet("/reports");
}

export async function downloadReport(report) {
  const blob = await apiDownloadBlob(`/reports/${report.id}/download`, { timeoutMs: 60000 });
  const url = URL.createObjectURL(blob);
  window.open(url, "_blank", "noopener,noreferrer");
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
}
