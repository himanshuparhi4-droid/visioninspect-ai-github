import { apiGet, apiPatch } from "./api";

export function getModelMetrics() {
  return apiGet("/model/metrics");
}

export function updateModelSettings(payload) {
  return apiPatch("/model/settings", payload);
}
