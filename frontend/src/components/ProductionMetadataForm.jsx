"use client";

export const EMPTY_METADATA = {
  batch_number: "",
  product_id: "",
  production_line: "",
  shift: "",
  operator_name: "",
  source_label: "",
  category: "",
};

export const EMPTY_CATALOG = {
  products: [],
  production_lines: [],
  batches: [],
  shifts: [],
};

export default function ProductionMetadataForm({
  value,
  catalog = EMPTY_CATALOG,
  onChange,
  disabled = false,
  placeholders = {},
  modelCategories = [],
}) {
  const metadata = { ...EMPTY_METADATA, ...value };
  const allProducts = catalog.products || [];
  const products = metadata.category ? allProducts.filter((item) => item.category === metadata.category) : allProducts;
  const productIds = new Set(products.map((item) => item.product_id));
  const lines = catalog.production_lines || [];
  const batches = metadata.category
    ? (catalog.batches || []).filter((item) => productIds.has(item.product_id))
    : catalog.batches || [];
  const shifts = catalog.shifts || [];
  const labels = {
    batch: "Select batch",
    product: "Select product",
    line: "Select line",
    shift: "Unassigned",
    ...placeholders,
  };
  const hasBatch = batches.some((item) => item.batch_number === metadata.batch_number);
  const hasProduct = products.some((item) => item.product_id === metadata.product_id);
  const hasLine = lines.some((item) => item.line_id === metadata.production_line);
  const hasShift = shifts.includes(metadata.shift);

  function update(field, nextValue) {
    if (field === "category") {
      onChange({ ...metadata, category: nextValue, product_id: "", batch_number: "" });
      return;
    }
    onChange({ ...metadata, [field]: nextValue });
  }

  function selectBatch(batchNumber) {
    const batch = batches.find((item) => item.batch_number === batchNumber);
    onChange({
      ...metadata,
      batch_number: batchNumber,
      product_id: batch?.product_id || metadata.product_id,
      production_line: batch?.production_line || metadata.production_line,
      shift: batch?.shift || metadata.shift,
    });
  }

  return (
    <div className="metadata-grid">
      <label>
        Inspection category
        <select
          value={metadata.category}
          onChange={(event) => update("category", event.target.value)}
          disabled={disabled}
        >
          <option value="">Select a category...</option>
          {modelCategories
            .filter((item) => item.available || item.runnable || item.trained || item.is_trained)
            .map((item) => (
              <option key={item.category} value={item.category}>
                {item.category.replaceAll("_", " ")} — {item.deployment_tier === "advanced" ? "Advanced" : "Portable"}
              </option>
            ))}
          {!modelCategories.length ? <option value="">Loading categories...</option> : null}
        </select>
      </label>
      <label>
        Batch number
        <select value={metadata.batch_number} onChange={(event) => selectBatch(event.target.value)} disabled={disabled}>
          <option value="">{labels.batch}</option>
          {metadata.batch_number && !hasBatch ? (
            <option value={metadata.batch_number}>{metadata.batch_number}</option>
          ) : null}
          {batches.map((batch) => (
            <option key={batch.batch_number} value={batch.batch_number}>
              {batch.batch_number}
            </option>
          ))}
        </select>
      </label>
      <label>
        Product ID
        <select
          value={metadata.product_id}
          onChange={(event) => update("product_id", event.target.value)}
          disabled={disabled}
        >
          <option value="">{labels.product}</option>
          {metadata.product_id && !hasProduct ? (
            <option value={metadata.product_id}>{metadata.product_id}</option>
          ) : null}
          {products.map((product) => (
            <option key={product.product_id} value={product.product_id}>
              {product.product_id} - {product.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Production line
        <select
          value={metadata.production_line}
          onChange={(event) => update("production_line", event.target.value)}
          disabled={disabled}
        >
          <option value="">{labels.line}</option>
          {metadata.production_line && !hasLine ? (
            <option value={metadata.production_line}>{metadata.production_line}</option>
          ) : null}
          {lines.map((line) => (
            <option key={line.line_id} value={line.line_id}>
              {line.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Shift
        <select value={metadata.shift} onChange={(event) => update("shift", event.target.value)} disabled={disabled}>
          <option value="">{labels.shift}</option>
          {metadata.shift && !hasShift ? <option value={metadata.shift}>{metadata.shift}</option> : null}
          {shifts.map((shift) => (
            <option key={shift} value={shift}>
              {shift}
            </option>
          ))}
        </select>
      </label>
      <label>
        Operator
        <input
          value={metadata.operator_name}
          onChange={(event) => update("operator_name", event.target.value)}
          disabled={disabled}
        />
      </label>
      <label>
        Source label
        <input
          value={metadata.source_label}
          onChange={(event) => update("source_label", event.target.value)}
          disabled={disabled}
        />
      </label>
    </div>
  );
}
