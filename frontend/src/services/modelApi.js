import { apiGet, apiPatch } from "./api";

export function getModelMetrics() {
  return apiGet("/model/metrics");
}

export function getModelSettings() {
  return apiGet("/model/settings");
}

export function updateModelSettings(payload) {
  return apiPatch("/model/settings", payload);
}
