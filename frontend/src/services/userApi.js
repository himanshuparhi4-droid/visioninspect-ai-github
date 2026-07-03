import { apiGet, apiPatch, apiPost } from "./api";

export function listUsers() {
  return apiGet("/users");
}

export function createUser(payload) {
  return apiPost("/users", payload);
}

export function updateUser(userId, payload) {
  return apiPatch(`/users/${userId}`, payload);
}

export function resetUserPassword(userId, password) {
  return apiPost(`/users/${userId}/reset-password`, { password });
}

export function approveUser(userId) {
  return apiPost(`/users/${userId}/approve`, {});
}

export function rejectUser(userId) {
  return apiPost(`/users/${userId}/reject`, {});
}

export function listAuditLogs({ limit = 50 } = {}) {
  return apiGet(`/audit-logs?limit=${limit}`);
}
