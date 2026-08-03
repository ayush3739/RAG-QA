const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, 'src/landing/components');

function walkDir(currentPath) {
  const files = fs.readdirSync(currentPath);
  for (const file of files) {
    const fullPath = path.join(currentPath, file);
    if (fs.statSync(fullPath).isDirectory()) {
      walkDir(fullPath);
    } else if (fullPath.endsWith('.tsx') || fullPath.endsWith('.ts')) {
      let content = fs.readFileSync(fullPath, 'utf8');
      
      // Revert instances of false /* forced animations */ to useReducedMotion()
      content = content.replace(/false \/\* forced animations \*\//g, 'useReducedMotion()');
      
      fs.writeFileSync(fullPath, content);
      console.log(`Reverted ${file}`);
    }
  }
}

walkDir(dir);
console.log('Revert done.');
