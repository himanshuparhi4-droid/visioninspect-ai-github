import { apiGet, apiPatch, apiPost } from "./api";

function appendMetadata(formData, metadata = {}) {
  Object.entries(metadata).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      formData.append(key, value);
    }
  });
  return formData;
}

function imageFormData(file, metadata) {
  const formData = new FormData();
  formData.append("file", file);
  return appendMetadata(formData, metadata);
}

function batchFormData(files, metadata) {
  const formData = new FormData();
  Array.from(files).forEach((file) => {
    formData.append("files", file);
  });
  return appendMetadata(formData, metadata);
}

export function inspectImage(file, metadata = {}) {
  return apiPost("/inspections/inspect", imageFormData(file, metadata), { timeoutMs: 120000 });
}

export function inspectBatch(files, metadata = {}) {
  return apiPost("/inspections/batch-inspect", batchFormData(files, metadata), { timeoutMs: 300000 });
}

export function listInspections({ skip = 0, limit = 50, productId = "", productionLine = "", reviewStatus = "" } = {}) {
  const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
  if (productId) params.set("product_id", productId);
  if (productionLine) params.set("production_line", productionLine);
  if (reviewStatus) params.set("review_status", reviewStatus);
  return apiGet(`/inspections?${params.toString()}`);
}

export function updateReviewStatus(inspectionId, reviewStatus, reviewNotes = "") {
  return apiPatch(`/inspections/${inspectionId}/review-status`, {
    review_status: reviewStatus,
    review_notes: reviewNotes,
  });
}

export function updateInspectionMetadata(inspectionId, metadata = {}) {
  return apiPatch(`/inspections/${inspectionId}/metadata`, metadata);
}

export function getCameraSamples(category = "") {
  return apiGet(`/inspections/camera-samples${category ? `?category=${category}` : ""}`);
}

export function getModelCategories() {
  return apiGet("/inspections/model-categories");
}

export function simulateCameraInspection({ frameIndex = 0, label = "", category = "" } = {}) {
  const params = new URLSearchParams({ frame_index: String(frameIndex) });
  if (label) params.set("label", label);
  if (category) params.set("category", category);
  return apiPost(`/inspections/camera-simulate?${params.toString()}`, {}, { timeoutMs: 120000 });
}
