import { apiGet, apiPost } from "./api";

export function getProductionCatalog() {
  return apiGet("/production/catalog");
}

export function createProduct(payload) {
  return apiPost("/production/products", payload);
}

export function createProductionLine(payload) {
  return apiPost("/production/lines", payload);
}

export function createBatchRecord(payload) {
  return apiPost("/production/batches", payload);
}
