import "./globals.css";

export const metadata = {
  title: "VisionInspect AI",
  description: "Manufacturing defect detection dashboard",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-scroll-behavior="smooth" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
