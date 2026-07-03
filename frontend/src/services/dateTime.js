function normalizeApiDate(value) {
  if (!value) return null;
  if (value instanceof Date) return value;
  const text = String(value);
  const hasTimezone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(text);
  return new Date(hasTimezone ? text : `${text}Z`);
}

export function formatDateTime(value) {
  const date = normalizeApiDate(value);
  if (!date || Number.isNaN(date.getTime())) return "Pending";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

export function formatTime(value) {
  const date = normalizeApiDate(value);
  if (!date || Number.isNaN(date.getTime())) return "Pending";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}
