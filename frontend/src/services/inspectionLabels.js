const REVIEW_STATUS_LABELS = {
  uploaded: "Uploaded",
  ai_pending: "AI Pending",
  ai_completed: "AI Completed",
  manual_review: "Manual Review",
  approved: "Approved",
  rejected: "Rejected",
  sent_for_rework: "Sent For Rework",
  re_inspected: "Re-inspected",
};

const SOURCE_TYPE_LABELS = {
  manual_upload: "Manual Upload",
  camera_simulation: "Camera Simulation",
};

export function formatReviewStatus(value) {
  return REVIEW_STATUS_LABELS[value] || value || "Pending";
}

export function formatSourceType(value) {
  return SOURCE_TYPE_LABELS[value] || value || "Unknown";
}
