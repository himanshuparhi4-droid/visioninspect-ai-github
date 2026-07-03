import { apiGet, getAuthToken } from "./api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export function getAnalyticsSummary(filters = {}) {
  const params = new URLSearchParams();
  if (filters.dateFrom) params.set("date_from", filters.dateFrom);
  if (filters.dateTo) params.set("date_to", filters.dateTo);
  if (filters.productionLine) params.set("production_line", filters.productionLine);
  if (filters.productId) params.set("product_id", filters.productId);
  const query = params.toString();
  return apiGet(`/analytics/summary${query ? `?${query}` : ""}`);
}

export async function downloadAnalyticsCsv() {
  const response = await fetch(`${API_BASE_URL}/analytics/export.csv`, {
    headers: {
      Authorization: `Bearer ${getAuthToken()}`,
    },
  });
  if (!response.ok) {
    throw new Error("Could not export analytics CSV");
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "visioninspect_inspections.csv";
  link.click();
  URL.revokeObjectURL(url);
}
