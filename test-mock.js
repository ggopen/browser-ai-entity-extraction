// Test the application end-to-end using the mock tileset.
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/root/.cache/ms-playwright/chrome-linux/chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1
  });
  const page = await ctx.newPage();
  const errors = [];
  const consoleMsgs = [];
  page.on('console', (msg) => {
    consoleMsgs.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on('pageerror', (err) => {
    errors.push(`PAGE ERROR: ${err.message}`);
  });

  console.log('=== Loading mock test page ===');
  await page.goto('http://localhost:8000/mock-test.html', { waitUntil: 'load' });
  let done = false;
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(500);
    const txt = await page.evaluate(() => document.getElementById('full').textContent);
    if (txt.includes('passed') || txt.includes('failed')) {
      done = true;
      break;
    }
  }
  const fullText = await page.evaluate(() => document.getElementById('full').textContent);
  console.log('--- Mock test log ---');
  console.log(fullText);
  console.log('--- Errors ---');
  errors.forEach((e) => console.log(e));

  await browser.close();
  console.log('=== Mock test done ===');
})().catch((e) => {
  console.error('Test failed:', e);
  process.exit(1);
});
