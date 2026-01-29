const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.join(process.env.HOME, 'VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx'));

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'BioInsight AI Team';
  pptx.title = 'BioInsight AI - RNA-seq Analysis Platform';

  const slidesDir = path.join(__dirname, 'slides');
  const slides = [
    'slide01_title.html',
    'slide02_overview.html',
    'slide03_pipeline.html',
    'slide04_deg_network.html',
    'slide05_pathway_validation.html',
    'slide06_ml_prediction.html',
    'slide07_singlecell.html',
    'slide08_techstack.html',
    'slide09_report.html',
    'slide10_results.html',
    'slide11_future.html',
    'slide12_conclusion.html',
  ];

  for (const slideFile of slides) {
    const htmlPath = path.join(slidesDir, slideFile);
    console.log(`Processing: ${slideFile}`);
    await html2pptx(htmlPath, pptx);
  }

  const outputPath = path.join(__dirname, 'BioInsight_RNAseq_Presentation.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`Presentation saved to: ${outputPath}`);
}

createPresentation().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
