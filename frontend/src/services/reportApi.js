import { apiGet, apiPost, getAuthToken } from "./api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export function createInspectionReport(inspectionId) {
  return apiPost(`/reports/inspection/${inspectionId}`, {});
}

export function listReports() {
  return apiGet("/reports");
}

export function getReport(reportId) {
  return apiGet(`/reports/${reportId}`);
}

export async function downloadReport(report) {
  const response = await fetch(`${API_BASE_URL}/reports/${report.id}/download`, {
    headers: {
      Authorization: `Bearer ${getAuthToken()}`,
    },
  });
  if (!response.ok) {
    throw new Error("Could not open report PDF");
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  window.open(url, "_blank", "noopener,noreferrer");
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
}
