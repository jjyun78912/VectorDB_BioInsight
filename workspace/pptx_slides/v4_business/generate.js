const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('/Users/admin/VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'BioInsight AI';
    pptx.title = 'BioInsight AI - Business Value';

    const baseDir = '/Users/admin/VectorDB_BioInsight/workspace/pptx_slides/v4_business';
    const slides = [
        'slide01_title.html', 'slide02_problem.html', 'slide03_values.html',
        'slide04_impact.html', 'slide05_features.html', 'slide06_workflow.html',
        'slide07_target.html', 'slide08_roadmap.html', 'slide09_summary.html'
    ];

    for (const slideFile of slides) {
        console.log(`Processing: ${slideFile}`);
        await html2pptx(path.join(baseDir, slideFile), pptx);
    }

    await pptx.writeFile({ fileName: path.join(baseDir, 'BioInsight_Business.pptx') });
    console.log('Created: BioInsight_Business.pptx');
}

createPresentation().catch(console.error);
