const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('/Users/admin/VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'BioInsight AI';
    pptx.title = 'BioInsight AI - Demo and Results';

    const baseDir = '/Users/admin/VectorDB_BioInsight/workspace/pptx_slides/v5_demo';
    const slides = [
        'slide01_title.html', 'slide02_dataset.html', 'slide03_benchmark.html',
        'slide04_deg.html', 'slide05_network.html', 'slide06_pathway.html',
        'slide07_ml.html', 'slide08_output.html', 'slide09_summary.html'
    ];

    for (const slideFile of slides) {
        console.log(`Processing: ${slideFile}`);
        await html2pptx(path.join(baseDir, slideFile), pptx);
    }

    await pptx.writeFile({ fileName: path.join(baseDir, 'BioInsight_Demo.pptx') });
    console.log('Created: BioInsight_Demo.pptx');
}

createPresentation().catch(console.error);
