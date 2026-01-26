#!/usr/bin/env python3
"""Update HTML report with new paper recommendations."""

import json
from pathlib import Path
import re

run_dir = Path('rnaseq_test_results/external_GSE81089_lung_cancer/run_20260126_113734')

# Load recommended_papers
with open(run_dir / 'recommended_papers.json', 'r') as f:
    papers_data = json.load(f)

# Read original report
report_path = run_dir / 'agent6_report' / 'report.html'
with open(report_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Generate new recommended papers section
classic_papers = papers_data.get('classic_papers', [])
breakthrough_papers = papers_data.get('breakthrough_papers', [])
other_papers = papers_data.get('other_papers', [])

def build_paper_card(paper, idx, paper_type):
    title = paper.get('title', 'No title')
    authors = paper.get('authors', 'Unknown')
    journal = paper.get('journal', '')
    year = paper.get('year', '')
    abstract = paper.get('abstract', '')[:300]
    if len(paper.get('abstract', '')) > 300:
        abstract += '...'
    pmid = paper.get('pmid', '')
    pubmed_url = paper.get('pubmed_url', f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/')
    citation_count = paper.get('citation_count', 0)
    citation_velocity = paper.get('citation_velocity', 0)
    relevance = paper.get('relevance_reason', '')

    if paper_type == "classic":
        type_badge = '<span class="paper-type-badge classic">📚 Classic Study</span>'
        type_class = "classic"
    elif paper_type == "breakthrough":
        type_badge = '<span class="paper-type-badge breakthrough">🚀 Emerging Research</span>'
        type_class = "breakthrough"
    else:
        type_badge = '<span class="paper-type-badge related">📄 Related Paper</span>'
        type_class = "related"

    citation_html = ''
    if citation_count > 0:
        citation_html = f'''
        <div class="citation-metrics">
            <span class="citation-count" title="Total citations">📊 인용: {citation_count:,}회</span>
            {f'<span class="citation-velocity" title="Citations per year">({citation_velocity:.1f}회/년)</span>' if citation_velocity > 0 else ''}
        </div>
        '''

    return f'''
    <div class="paper-card {type_class}">
        <div class="paper-number">{idx}</div>
        <div class="paper-content">
            {type_badge}
            <h4 class="paper-title">
                <a href="{pubmed_url}" target="_blank" rel="noopener">{title}</a>
            </h4>
            <p class="paper-meta">
                <span class="authors">{authors}</span>
                <span class="journal">{journal}</span>
                <span class="year">({year})</span>
            </p>
            {citation_html}
            <p class="paper-abstract">{abstract}</p>
            <div class="paper-footer">
                <span class="relevance-tag">{relevance}</span>
                <span class="pmid">PMID: <a href="{pubmed_url}" target="_blank">{pmid}</a></span>
            </div>
        </div>
    </div>
    '''

# Build cards
classic_cards = ''.join([build_paper_card(p, i, "classic") for i, p in enumerate(classic_papers[:3], 1)])
breakthrough_cards = ''.join([build_paper_card(p, i, "breakthrough") for i, p in enumerate(breakthrough_papers[:3], 1)])
other_cards = ''.join([build_paper_card(p, i, "related") for i, p in enumerate(other_papers[:3], 1)])

cancer_type = papers_data.get('cancer_type', 'cancer')
search_genes = papers_data.get('search_genes', [])

new_section = f'''
<section class="recommended-papers-section" id="recommended-papers">
    <h2>8.4 추천 논문</h2>

    <div class="papers-intro">
        <p>아래 논문들은 <strong>{cancer_type}</strong> 및 분석에서 도출된 주요 유전자
        ({', '.join(search_genes[:5])})를 기반으로 PubMed/Semantic Scholar에서 검색 및 평가된 결과입니다.
        인용 지표와 학술적 영향력을 기준으로 선정되었습니다.</p>
        <div class="papers-stats">
            <span class="stat-item"><span class="stat-icon">📚</span> Classic: {len(classic_papers)}편</span>
            <span class="stat-item"><span class="stat-icon">🚀</span> Emerging: {len(breakthrough_papers)}편</span>
            <span class="stat-item"><span class="stat-icon">📄</span> Related: {len(other_papers)}편</span>
        </div>
    </div>

    <div class="paper-category">
        <h3 class="category-title">📚 필수 참고 논문 (Classic Studies)</h3>
        <p class="category-desc">해당 분야의 기초가 되는 고인용 논문들입니다. (50회 이상 인용, 3년 이상 경과)</p>
        <div class="paper-list">
            {classic_cards if classic_cards else '<p class="no-papers">인용 데이터 기준을 만족하는 Classic 논문이 없습니다.</p>'}
        </div>
    </div>

    <div class="paper-category">
        <h3 class="category-title">🚀 최신 주목 논문 (Emerging Research)</h3>
        <p class="category-desc">빠르게 인용되고 있는 최근 연구들입니다. (10회 이상 인용, 높은 인용 속도)</p>
        <div class="paper-list">
            {breakthrough_cards if breakthrough_cards else '<p class="no-papers">인용 데이터 기준을 만족하는 Breakthrough 논문이 없습니다.</p>'}
        </div>
    </div>

    <div class="paper-category">
        <h3 class="category-title">📄 관련 논문 (Related Papers)</h3>
        <p class="category-desc">분석 유전자와 관련된 최신 논문들입니다. (인용 데이터 미확인 또는 기준 미달)</p>
        <div class="paper-list">
            {other_cards if other_cards else '<p class="no-papers">관련 논문이 없습니다.</p>'}
        </div>
    </div>

    <div class="papers-disclaimer">
        <p><strong>참고:</strong> 논문 분류는 인용수와 출판연도를 기반으로 자동 산정되었습니다.
        Classic Study는 3년 이상 경과 및 50회 이상 인용된 논문,
        Emerging Research는 2년 이내 출판되어 10회 이상 인용된 논문입니다.</p>
    </div>
</section>

<style>
.paper-type-badge.related {{
    background: #e3f2fd;
    color: #1565c0;
}}
.paper-card.related {{
    border-left: 4px solid #2196F3;
}}
</style>
'''

# Replace the recommended-papers section
pattern = r'<section class="recommended-papers-section" id="recommended-papers">.*?</section>'
html_updated = re.sub(pattern, new_section, html, flags=re.DOTALL)

# Save updated report
updated_path = run_dir / 'agent6_report' / 'report_updated.html'
with open(updated_path, 'w', encoding='utf-8') as f:
    f.write(html_updated)

print(f"Updated report saved: {updated_path}")
print(f"\nPaper counts:")
print(f"  Classic: {len(classic_papers)}")
print(f"  Breakthrough: {len(breakthrough_papers)}")
print(f"  Related (Other): {len(other_papers)}")
