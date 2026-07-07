import { BarChart3, CheckCircle2, ClipboardCheck, Gauge, ScanSearch, TriangleAlert, XCircle } from "lucide-react";

const cards = [
  { key: "total_inspections", label: "Total", icon: ScanSearch, tone: "blue", hint: "All inspected images" },
  { key: "defective_count", label: "Defective", icon: XCircle, tone: "red", hint: "Needs attention" },
  { key: "good_count", label: "Good", icon: CheckCircle2, tone: "green", hint: "Passed by AI" },
  { key: "critical_count", label: "Critical", icon: TriangleAlert, tone: "red", hint: "Highest severity" },
  { key: "rework_queue", label: "Rework Queue", icon: ClipboardCheck, tone: "amber", hint: "Open repair items" },
  { key: "fail_count", label: "Failed", icon: TriangleAlert, tone: "red", hint: "Rejected products" },
  {
    key: "average_confidence",
    label: "Avg Confidence",
    icon: Gauge,
    percent: true,
    tone: "blue",
    hint: "Model certainty",
  },
  {
    key: "defect_rate",
    label: "Defect Rate",
    icon: BarChart3,
    percent: true,
    tone: "purple",
    hint: "Defects vs total",
  },
];

export default function AnalyticsCards({ summary }) {
  return (
    <section className="stats-grid">
      {cards.map((card) => {
        const Icon = card.icon;
        const raw = summary?.[card.key] ?? 0;
        const value = card.percent ? `${(raw * 100).toFixed(1)}%` : raw;
        return (
          <div key={card.key} className={`stat-card stat-${card.tone}`}>
            <span className="stat-icon">
              <Icon size={18} />
            </span>
            <div>
              <small>{card.label}</small>
              <strong>{value}</strong>
              <em>{card.hint}</em>
            </div>
          </div>
        );
      })}
    </section>
  );
}
