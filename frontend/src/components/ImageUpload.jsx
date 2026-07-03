"use client";

import { ImageUp, Loader2, ScanSearch } from "lucide-react";

export default function ImageUpload({ file, onFileChange, onInspect, loading }) {
  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>Image Acquisition</h2>
          <p>Upload one product image for AI inspection.</p>
        </div>
        <ImageUp size={22} />
      </div>

      <label className="dropzone">
        <input
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff,image/webp"
          onChange={(event) => onFileChange(event.target.files?.[0] || null)}
        />
        <span className="dropzone-icon">
          <ImageUp size={26} />
        </span>
        <strong>{file ? file.name : "Choose inspection image"}</strong>
        <small>PNG, JPG, BMP, TIFF, or WebP</small>
      </label>

      <button className="primary-button" type="button" onClick={onInspect} disabled={!file || loading}>
        {loading ? <Loader2 className="spin" size={18} /> : <ScanSearch size={18} />}
        <span>{loading ? "Inspecting" : "Run Inspection"}</span>
      </button>
    </section>
  );
}
