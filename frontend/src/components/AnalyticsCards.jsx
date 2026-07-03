import { BarChart3, CheckCircle2, ClipboardCheck, Gauge, ScanSearch, TriangleAlert, XCircle } from "lucide-react";

const cards = [
  { key: "total_inspections", label: "Total", icon: ScanSearch },
  { key: "defective_count", label: "Defective", icon: XCircle },
  { key: "good_count", label: "Good", icon: CheckCircle2 },
  { key: "critical_count", label: "Critical", icon: TriangleAlert },
  { key: "rework_queue", label: "Rework Queue", icon: ClipboardCheck },
  { key: "fail_count", label: "Failed", icon: TriangleAlert },
  { key: "average_confidence", label: "Avg Confidence", icon: Gauge, percent: true },
  { key: "defect_rate", label: "Defect Rate", icon: BarChart3, percent: true },
];

export default function AnalyticsCards({ summary }) {
  return (
    <section className="stats-grid">
      {cards.map((card) => {
        const Icon = card.icon;
        const raw = summary?.[card.key] ?? 0;
        const value = card.percent ? `${(raw * 100).toFixed(1)}%` : raw;
        return (
          <div key={card.key} className="stat-card">
            <span className="stat-icon">
              <Icon size={18} />
            </span>
            <small>{card.label}</small>
            <strong>{value}</strong>
          </div>
        );
      })}
    </section>
  );
}
