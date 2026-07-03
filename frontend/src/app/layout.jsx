import Script from "next/script";

import "./globals.css";

const stripExtensionAttributesScript = `
(function () {
  function shouldRemoveAttribute(name) {
    return name.indexOf("bis_") === 0 || name.indexOf("__processed_") === 0;
  }

  function cleanNode(node) {
    if (!node || node.nodeType !== 1 || !node.attributes) return;
    for (var index = node.attributes.length - 1; index >= 0; index -= 1) {
      var name = node.attributes[index].name;
      if (shouldRemoveAttribute(name)) {
        node.removeAttribute(name);
      }
    }
  }

  function cleanTree(root) {
    cleanNode(root);
    if (!root || !root.querySelectorAll) return;
    var nodes = root.querySelectorAll("*");
    for (var index = 0; index < nodes.length; index += 1) {
      cleanNode(nodes[index]);
    }
  }

  cleanTree(document.documentElement);

  if (typeof MutationObserver !== "undefined") {
    new MutationObserver(function (mutations) {
      for (var index = 0; index < mutations.length; index += 1) {
        cleanNode(mutations[index].target);
        for (var nodeIndex = 0; nodeIndex < mutations[index].addedNodes.length; nodeIndex += 1) {
          cleanTree(mutations[index].addedNodes[nodeIndex]);
        }
      }
    }).observe(document.documentElement, {
      attributes: true,
      childList: true,
      subtree: true
    });
  }
})();
`;

export const metadata = {
  title: "VisionInspect AI",
  description: "Manufacturing defect detection dashboard"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <Script
          id="strip-extension-attrs"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{ __html: stripExtensionAttributesScript }}
        />
        {children}
      </body>
    </html>
  );
}
