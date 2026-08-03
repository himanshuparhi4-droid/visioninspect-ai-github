import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

export default defineConfig([
  ...nextVitals,
  {
    files: ["src/**/*.{js,jsx}"],
    rules: {
      // Pages intentionally perform mount-only API loading and synchronize
      // related form state when the selected inspection changes.
      "react-hooks/exhaustive-deps": "off",
      "react-hooks/set-state-in-effect": "off",
      // Inspection images use authenticated, local, or Cloudinary URLs whose
      // dimensions are not known at build time.
      "@next/next/no-img-element": "off",
    },
  },
  globalIgnores([".next/**", "out/**", "node_modules/**"]),
]);
