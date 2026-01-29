const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.join(process.env.HOME, 'VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx'));

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'BioInsight AI Team';
  pptx.title = 'BioInsight AI - DB 검증 & RAG 리포트';

  const slides = [
    'slide_agent4_validation.html',
    'slide_agent6_rag_report.html',
  ];

  for (const slide of slides) {
    const htmlPath = path.join(__dirname, slide);
    console.log(`Processing: ${slide}`);
    await html2pptx(htmlPath, pptx);
  }

  const outputPath = path.join(__dirname, 'Validation_RAG.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`Presentation saved to: ${outputPath}`);
}

createPresentation().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
