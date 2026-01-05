const pptxgen = require('pptxgenjs');
const path = require('path');
const fs = require('fs');

// Patch require to use local node_modules for html2pptx dependencies
const originalRequire = module.constructor.prototype.require;
const localNodeModules = path.join(__dirname, 'node_modules');
module.constructor.prototype.require = function(id) {
    if (id === 'playwright' || id === 'pptxgenjs' || id === 'sharp') {
        try {
            return originalRequire.call(this, path.join(localNodeModules, id));
        } catch (e) {
            // fallback to original
        }
    }
    return originalRequire.call(this, id);
};

const html2pptx = require('/Users/admin/VectorDB_BioInsight/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'BioInsight AI';
    pptx.title = 'RNA-seq Cancer Analysis Pipeline - Weekly Report';
    pptx.subject = 'Weekly Progress Report';

    const slidesDir = '/Users/admin/VectorDB_BioInsight/workspace/pptx_slides';

    const slides = [
        'slide1_title.html',
        'slide2_overview.html',
        'slide3_results.html',
        'slide4_hub_genes.html',
        'slide5_rag.html',
        'slide6_database.html',
        'slide7_pipeline.html',
        'slide8_next.html',
        'slide9_end.html',
        'slide10_qna.html'
    ];

    for (const slideFile of slides) {
        const slidePath = path.join(slidesDir, slideFile);
        console.log(`Processing: ${slideFile}`);
        try {
            await html2pptx(slidePath, pptx);
            console.log(`  ✓ ${slideFile} converted`);
        } catch (err) {
            console.error(`  ✗ Error with ${slideFile}:`, err.message);
        }
    }

    const outputPath = '/Users/admin/VectorDB_BioInsight/workspace/rnaseq_weekly_report.pptx';
    await pptx.writeFile({ fileName: outputPath });
    console.log(`\n✓ Presentation saved to: ${outputPath}`);
}

createPresentation().catch(console.error);
