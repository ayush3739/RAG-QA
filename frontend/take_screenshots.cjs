const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const screenshotsDir = path.join(__dirname, '..', 'screenshots');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  console.log('Launching browser...');
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1536, height: 730 });

  console.log('Navigating to http://localhost:3000 ...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 2000)); // wait for init

  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'library', label: 'Library' },
    { id: 'conversations', label: 'Conversations' },
    { id: 'research', label: 'Research Workspace' },
    { id: 'sessions', label: 'Sessions Manager' },
    { id: 'settings', label: 'Settings' }
  ];

  async function clickTab(label) {
    await page.evaluate((lbl) => {
      const buttons = Array.from(document.querySelectorAll('nav button'));
      const target = buttons.find(b => b.textContent.includes(lbl));
      if (target) target.click();
    }, label);
    await new Promise(r => setTimeout(r, 1000));
  }

  async function clickThemeToggle() {
    await page.evaluate(() => {
      const btn = document.querySelector('button[title="Toggle Theme"]');
      if (btn) btn.click();
    });
    await new Promise(r => setTimeout(r, 500));
  }
  
  // ensure we start in dark mode
  const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
  if (!isDark) {
    await clickThemeToggle();
  }

  for (const tab of tabs) {
    console.log(`Processing ${tab.label}...`);
    await clickTab(tab.label);

    // Dark mode screenshot
    await page.screenshot({ path: path.join(screenshotsDir, `${tab.id}_dark.png`) });
    console.log(`Saved ${tab.id}_dark.png`);

    // Toggle to light mode
    await clickThemeToggle();
    await page.screenshot({ path: path.join(screenshotsDir, `${tab.id}_light.png`) });
    console.log(`Saved ${tab.id}_light.png`);

    // Toggle back to dark mode
    await clickThemeToggle();
  }

  await browser.close();
  console.log('All screenshots taken! Check the /screenshots folder.');
})();
