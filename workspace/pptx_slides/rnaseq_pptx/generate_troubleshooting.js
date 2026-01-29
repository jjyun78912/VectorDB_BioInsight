const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.join(process.env.HOME, 'VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx'));

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'BioInsight AI Team';
  pptx.title = 'BioInsight AI - 트러블슈팅 가이드';

  const htmlPath = path.join(__dirname, 'slide_troubleshooting.html');
  console.log(`Processing: slide_troubleshooting.html`);
  await html2pptx(htmlPath, pptx);

  const outputPath = path.join(__dirname, 'Troubleshooting.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`Presentation saved to: ${outputPath}`);
}

createPresentation().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
