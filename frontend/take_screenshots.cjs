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

  async function clickTab(label) {
    await page.evaluate((lbl) => {
      const buttons = Array.from(document.querySelectorAll('nav button'));
      const target = buttons.find(b => b.textContent.includes(lbl));
      if (target) target.click();
    }, label);
    await new Promise(r => setTimeout(r, 1000));
  }

  async function clickNewChat() {
    await page.evaluate(() => {
      const btn = document.getElementById('btn-new-research');
      if (btn) btn.click();
    });
    await new Promise(r => setTimeout(r, 1000));
  }

  async function clickThemeToggle() {
    await page.evaluate(() => {
      const btn = document.querySelector('button[title="Toggle Theme"]');
      if (btn) btn.click();
    });
    await new Promise(r => setTimeout(r, 500));
  }
  
  // Ensure we start in dark mode
  const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
  if (!isDark) {
    await clickThemeToggle();
  }

  // --- DARK MODE ---
  console.log('Taking Dark mode screenshots...');
  
  // Dashboard
  await clickTab('Dashboard');
  await page.screenshot({ path: path.join(screenshotsDir, 'dashboard_dark.png') });
  console.log('Saved dashboard_dark.png');

  // Chats
  await clickTab('Chats');
  await page.screenshot({ path: path.join(screenshotsDir, 'chats_dark.png') });
  console.log('Saved chats_dark.png');

  // New Chat
  await clickNewChat();
  await page.screenshot({ path: path.join(screenshotsDir, 'new_chat_dark.png') });
  console.log('Saved new_chat_dark.png');

  // Documents
  await clickTab('Documents');
  await page.screenshot({ path: path.join(screenshotsDir, 'documents_dark.png') });
  console.log('Saved documents_dark.png');

  // Settings
  await clickTab('Settings');
  await page.screenshot({ path: path.join(screenshotsDir, 'settings_dark.png') });
  console.log('Saved settings_dark.png');

  // --- LIGHT MODE ---
  console.log('Switching to Light mode...');
  await clickThemeToggle();

  console.log('Taking Light mode screenshots...');
  
  // Settings (already there)
  await page.screenshot({ path: path.join(screenshotsDir, 'settings_light.png') });
  console.log('Saved settings_light.png');
  
  // Dashboard
  await clickTab('Dashboard');
  await page.screenshot({ path: path.join(screenshotsDir, 'dashboard_light.png') });
  console.log('Saved dashboard_light.png');

  // Chats
  await clickTab('Chats');
  await page.screenshot({ path: path.join(screenshotsDir, 'chats_light.png') });
  console.log('Saved chats_light.png');

  // New Chat
  await clickNewChat();
  await page.screenshot({ path: path.join(screenshotsDir, 'new_chat_light.png') });
  console.log('Saved new_chat_light.png');

  // Documents
  await clickTab('Documents');
  await page.screenshot({ path: path.join(screenshotsDir, 'documents_light.png') });
  console.log('Saved documents_light.png');

  // Revert back to Dark mode
  await clickThemeToggle();

  await browser.close();
  console.log('All screenshots taken successfully!');
})();
