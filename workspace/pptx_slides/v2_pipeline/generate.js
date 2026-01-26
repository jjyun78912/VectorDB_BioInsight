const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('/Users/admin/VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'BioInsight AI';
    pptx.title = 'BioInsight AI - RNA-seq Pipeline';

    const baseDir = '/Users/admin/VectorDB_BioInsight/workspace/pptx_slides/v2_pipeline';
    const slides = [
        'slide01_title.html', 'slide02_overview.html', 'slide03_6agent.html',
        'slide04_agent1.html', 'slide05_agent2.html', 'slide06_agent34.html',
        'slide07_agent56.html', 'slide08_singlecell.html', 'slide09_benchmark.html',
        'slide10_summary.html'
    ];

    for (const slideFile of slides) {
        console.log(`Processing: ${slideFile}`);
        await html2pptx(path.join(baseDir, slideFile), pptx);
    }

    await pptx.writeFile({ fileName: path.join(baseDir, 'BioInsight_Pipeline.pptx') });
    console.log('Created: BioInsight_Pipeline.pptx');
}

createPresentation().catch(console.error);
