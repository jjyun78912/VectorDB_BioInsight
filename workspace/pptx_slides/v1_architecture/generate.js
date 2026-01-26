const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('/Users/admin/VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'BioInsight AI';
    pptx.title = 'BioInsight AI - Technical Architecture';
    pptx.subject = 'Technical Architecture Overview';

    const baseDir = '/Users/admin/VectorDB_BioInsight/workspace/pptx_slides/v1_architecture';
    const slides = [
        'slide01_title.html',
        'slide02_overview.html',
        'slide03_techstack.html',
        'slide04_frontend.html',
        'slide05_backend.html',
        'slide06_core.html',
        'slide07_data.html',
        'slide08_external.html',
        'slide09_llm.html',
        'slide10_summary.html'
    ];

    for (const slideFile of slides) {
        const htmlPath = path.join(baseDir, slideFile);
        console.log(`Processing: ${slideFile}`);
        await html2pptx(htmlPath, pptx);
    }

    const outputPath = path.join(baseDir, 'BioInsight_Architecture.pptx');
    await pptx.writeFile({ fileName: outputPath });
    console.log(`Created: ${outputPath}`);
}

createPresentation().catch(console.error);
