import { Flame } from "lucide-react";

export default function DefectHeatmap({ imageUrl }) {
  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>Defect Heatmap</h2>
          <p>Localized anomaly visualization.</p>
        </div>
        <Flame size={22} />
      </div>
      {imageUrl ? (
        <div className="image-frame">
          <img src={imageUrl} alt="Defect heatmap" />
        </div>
      ) : (
        <div className="empty-visual">Heatmap appears after inspection.</div>
      )}
    </section>
  );
}
