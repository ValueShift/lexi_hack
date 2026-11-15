// Pre-renders the math in pandoc's HTML with KaTeX and inlines KaTeX's CSS and fonts, so the page needs no
// JavaScript or network; also inlines local stylesheets and images.
// Usage: node katex_render.js in.html out.html <repo root>   (needs: npm install katex)
const fs = require("fs"), path = require("path");
const katex = require("katex");
const [inp, out, root] = process.argv.slice(2);
let html = fs.readFileSync(inp, "utf8");
const unesc = s => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&");
let n = 0;
html = html.replace(/<span class="math (inline|display)">\\[(\[]([\s\S]*?)\\[)\]]<\/span>/g, (_, kind, tex) => {
  n++;
  return katex.renderToString(unesc(tex), { displayMode: kind === "display", throwOnError: true, output: "html" });
});
html = html.replace(/<script[^>]*src="[^"]*(mathjax|polyfill)[^"]*"[^>]*>[\s\S]*?<\/script>\s*/gi, "");  // no remote scripts
html = html.replace(/<link rel="stylesheet" href="([^"]+)" \/>/g, (m, href) =>
  fs.existsSync(href) ? `<style>${fs.readFileSync(href, "utf8")}</style>` : m);
html = html.replace(/<img src="([^"]+\.(png|jpg|jpeg|svg))"/g, (m, src, ext) => {
  const f = path.resolve(root || ".", src);
  if (!fs.existsSync(f)) return m;
  const mime = ext === "svg" ? "image/svg+xml" : ext === "png" ? "image/png" : "image/jpeg";
  return `<img src="data:${mime};base64,${fs.readFileSync(f).toString("base64")}"`;
});
const dist = path.join(path.dirname(require.resolve("katex")), "dist") .replace(/dist[\/\\]dist$/, "dist");
const cssPath = fs.existsSync(path.join(dist, "katex.min.css")) ? path.join(dist, "katex.min.css") : require.resolve("katex/dist/katex.min.css");
let css = fs.readFileSync(cssPath, "utf8");
const fontDir = path.join(path.dirname(cssPath), "fonts");
// keep only the woff2 source of each @font-face and inline it
css = css.replace(/src:([^;}]*)/g, (m, srcs) => {
  const f = /url\((fonts\/[^)]+\.woff2)\)/.exec(srcs);
  if (!f) return m;
  const data = fs.readFileSync(path.join(path.dirname(cssPath), f[1])).toString("base64");
  return `src:url(data:font/woff2;base64,${data}) format("woff2")`;
});
html = html.replace("</head>", `<style>${css}\n.katex-display{overflow-x:auto;overflow-y:hidden;padding:4px 0}</style>\n</head>`);
fs.writeFileSync(out, html);
console.log(`katex: rendered ${n} formulas`);
