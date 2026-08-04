"use client";

function countBy(items, field, expected) {
  return items.filter((item) => item[field] === expected).length;
}

function confidenceText(value) {
  return value != null ? `${(value * 100).toFixed(1)}%` : "Pending";
}

export default function BatchInspectionResults({ results = [], summary = null, failures = [] }) {
  if (!results.length && !failures.length) return null;

  const cards = [
    ["Total", summary?.total ?? results.length],
    ["Good", summary?.good ?? countBy(results, "prediction", "Good")],
    ["Defective", summary?.defective ?? countBy(results, "prediction", "Defective")],
    ["QA Fail", summary?.fail ?? countBy(results, "pass_fail", "Fail")],
    ["Processing Errors", summary?.failed ?? failures.length],
    ["Critical", summary?.critical ?? countBy(results, "severity_level", "Critical")],
    ["Avg Confidence", confidenceText(summary?.average_confidence)],
  ];

  return (
    <>
      <div className="stats-grid compact-stats">
        {cards.map(([label, value]) => (
          <div className="stat-card" key={label}>
            <small>{label}</small>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Image</th>
              <th>Prediction</th>
              <th>Defect</th>
              <th>Severity</th>
              <th>Decision</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {results.map((item) => (
              <tr key={item.id}>
                <td>{item.source_label || item.original_filename || "Uploaded image"}</td>
                <td>{item.prediction}</td>
                <td>{item.defect_type}</td>
                <td>{item.severity_level}</td>
                <td>{item.pass_fail}</td>
                <td>{confidenceText(item.confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {failures.length ? (
        <section className="error-panel" role="status">
          <strong>{failures.length} images could not be inspected</strong>
          {failures.map((failure) => (
            <p key={`${failure.file_name}-${failure.status}`}>
              {failure.file_name}: {failure.message}
            </p>
          ))}
        </section>
      ) : null}
    </>
  );
}
