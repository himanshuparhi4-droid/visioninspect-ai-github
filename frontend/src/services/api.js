const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const TOKEN_KEY = "visioninspect_token";

export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.code = options.code;
    this.details = options.details;
    this.requestId = options.requestId;
  }
}

function formatValidationDetails(details) {
  if (!Array.isArray(details)) return "";
  return details
    .map((item) => {
      if (typeof item === "string") return item;
      const field = item.field ? `${item.field}: ` : "";
      return `${field}${item.message || "Invalid value"}`;
    })
    .join("; ");
}

function normalizeErrorPayload(payload) {
  if (typeof payload === "string") {
    return { message: payload || "API request failed" };
  }

  const structured = payload?.error;
  if (structured) {
    const validationText = formatValidationDetails(structured.details);
    return {
      message: validationText || structured.message || "API request failed",
      code: structured.code,
      details: structured.details,
      requestId: structured.request_id,
    };
  }

  if (Array.isArray(payload?.detail)) {
    const validationText = formatValidationDetails(
      payload.detail.map((item) => ({
        field: Array.isArray(item.loc) ? item.loc.join(".") : item.loc,
        message: item.msg,
      }))
    );
    return { message: validationText || "Please check the submitted data and try again", details: payload.detail };
  }

  return {
    message: payload?.detail || payload?.message || "API request failed",
    details: payload,
  };
}

export function getAuthToken() {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token) {
  if (typeof window === "undefined") return;
  if (token) {
    window.localStorage.setItem(TOKEN_KEY, token);
  } else {
    window.localStorage.removeItem(TOKEN_KEY);
  }
}

export async function apiRequest(path, options = {}) {
  const token = Object.prototype.hasOwnProperty.call(options, "token") ? options.token : getAuthToken();
  const headers = new Headers(options.headers || {});

  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const normalized = normalizeErrorPayload(payload);
    throw new ApiError(normalized.message, {
      status: response.status,
      code: normalized.code,
      details: normalized.details,
      requestId: normalized.requestId || response.headers.get("x-request-id"),
    });
  }

  return payload;
}

export function apiGet(path, options = {}) {
  return apiRequest(path, { ...options, method: "GET" });
}

export function apiPost(path, body, options = {}) {
  return apiRequest(path, {
    ...options,
    method: "POST",
    body: body instanceof FormData ? body : JSON.stringify(body),
  });
}

export function apiPatch(path, body, options = {}) {
  return apiRequest(path, {
    ...options,
    method: "PATCH",
    body: body instanceof FormData ? body : JSON.stringify(body),
  });
}
