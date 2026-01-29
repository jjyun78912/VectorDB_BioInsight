const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.join(process.env.HOME, 'VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx'));

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'BioInsight AI Team';
  pptx.title = 'BioInsight AI - RNA-seq 분석 파이프라인';

  const htmlPath = path.join(__dirname, 'slide_pipeline_single.html');
  console.log(`Processing: slide_pipeline_single.html`);
  await html2pptx(htmlPath, pptx);

  const outputPath = path.join(__dirname, 'BioInsight_RNA-seq_Pipeline.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`Presentation saved to: ${outputPath}`);
}

createPresentation().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
