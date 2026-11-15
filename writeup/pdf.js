// Prints the HTML write-up to an A4 PDF with Chromium (Playwright), with page numbers in the footer.
// Usage: node pdf.js in.html out.pdf
const { chromium } = require("playwright");
const path = require("path");
(async () => {
  const [inp, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(inp), { waitUntil: "load" });
  await page.emulateMedia({ media: "print", colorScheme: "light" });
  await page.pdf({
    path: out, format: "A4", printBackground: true,
    margin: { top: "16mm", bottom: "18mm", left: "16mm", right: "16mm" },
    displayHeaderFooter: true, headerTemplate: "<span></span>",
    footerTemplate: '<div style="width:100%;font:8px -apple-system,Helvetica,Arial,sans-serif;color:#8a8984;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log("pdf: " + out);
})();
