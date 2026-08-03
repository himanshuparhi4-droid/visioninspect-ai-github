import { apiGet } from "./api";

export function getProductionCatalog() {
  return apiGet("/production/catalog");
}
