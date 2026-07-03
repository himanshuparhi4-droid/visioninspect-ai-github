export default function SeverityBadge({ level }) {
  const value = level || "Pending";
  const className = `severity-badge severity-${value.toLowerCase()}`;
  return <span className={className}>{value}</span>;
}
