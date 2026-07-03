import { apiGet, apiPost, setAuthToken } from "./api";

export async function registerUser(payload) {
  const result = await apiPost("/auth/register", payload, { token: null });
  return result;
}

export async function login(payload) {
  const result = await apiPost("/auth/login", payload, { token: null });
  setAuthToken(result.access_token);
  return result;
}

export async function getCurrentUser() {
  return apiGet("/auth/me");
}

export function logout() {
  setAuthToken(null);
}
