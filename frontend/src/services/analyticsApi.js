import { apiDownloadBlob, apiGet } from "./api";

function analyticsQuery(filters = {}) {
  const params = new URLSearchParams();
  if (filters.dateFrom) params.set("date_from", filters.dateFrom);
  if (filters.dateTo) params.set("date_to", filters.dateTo);
  if (filters.productionLine) params.set("production_line", filters.productionLine);
  if (filters.productId) params.set("product_id", filters.productId);
  if (filters.defectType) params.set("defect_type", filters.defectType);
  return params.toString();
}

export function getAnalyticsSummary(filters = {}) {
  const query = analyticsQuery(filters);
  return apiGet(`/analytics/summary${query ? `?${query}` : ""}`);
}

export async function downloadAnalyticsCsv(filters = {}) {
  const query = analyticsQuery(filters);
  const blob = await apiDownloadBlob(`/analytics/export.csv${query ? `?${query}` : ""}`, { timeoutMs: 60000 });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "visioninspect_inspections.csv";
  link.click();
  URL.revokeObjectURL(url);
}
