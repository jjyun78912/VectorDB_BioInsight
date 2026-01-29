const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.join(process.env.HOME, 'VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx'));

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'BioInsight AI Team';
  pptx.title = 'BioInsight AI - 프로젝트 목차';

  const htmlPath = path.join(__dirname, 'slides_ko', 'slide02_toc.html');
  console.log(`Processing: slide02_toc.html`);
  await html2pptx(htmlPath, pptx);

  const outputPath = path.join(__dirname, 'TOC.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`Presentation saved to: ${outputPath}`);
}

createPresentation().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
