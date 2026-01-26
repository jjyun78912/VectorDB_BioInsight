const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('/Users/admin/VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'BioInsight AI';
    pptx.title = 'BioInsight AI - ML/AI Features';

    const baseDir = '/Users/admin/VectorDB_BioInsight/workspace/pptx_slides/v3_ml_ai';
    const slides = [
        'slide01_title.html', 'slide02_overview.html', 'slide03_catboost.html',
        'slide04_shap.html', 'slide05_confusable.html', 'slide06_rag.html',
        'slide07_llm.html', 'slide08_dgidb.html', 'slide09_guardrails.html',
        'slide10_summary.html'
    ];

    for (const slideFile of slides) {
        console.log(`Processing: ${slideFile}`);
        await html2pptx(path.join(baseDir, slideFile), pptx);
    }

    await pptx.writeFile({ fileName: path.join(baseDir, 'BioInsight_ML_AI.pptx') });
    console.log('Created: BioInsight_ML_AI.pptx');
}

createPresentation().catch(console.error);
