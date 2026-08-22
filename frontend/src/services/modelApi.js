import { apiGet, apiPatch, apiPost } from "./api";

export function getModelMetrics() {
  return apiGet("/model/metrics");
}

export function updateModelSettings(payload) {
  return apiPatch("/model/settings", payload);
}

export function warmModelCategory(category) {
  return apiPost(`/model/warmup/${encodeURIComponent(category)}`, {}, { timeoutMs: 120000 });
}
