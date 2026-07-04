"use client";

function countBy(items, field, expected) {
  return items.filter((item) => item[field] === expected).length;
}

function confidenceText(value) {
  return value != null ? `${(value * 100).toFixed(1)}%` : "Pending";
}

export default function BatchInspectionResults({ results = [], summary = null }) {
  if (!results.length) return null;

  const cards = [
    ["Total", summary?.total ?? results.length],
    ["Good", summary?.good ?? countBy(results, "prediction", "Good")],
    ["Defective", summary?.defective ?? countBy(results, "prediction", "Defective")],
    ["Failed", summary?.fail ?? countBy(results, "pass_fail", "Fail")],
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
    </>
  );
}
