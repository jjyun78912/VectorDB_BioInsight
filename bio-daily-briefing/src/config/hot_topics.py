"""
Predefined Hot Topics for BIO Daily Briefing

업계가 주목하는 핫토픽 목록
분기별로 검토하여 업데이트 권장
"""

PREDEFINED_HOT_TOPICS = {
    # === GLP-1 / 비만치료 (통합 - 가장 핫한 키워드) ===
    "GLP-1/비만치료": {
        "keywords": [
            # GLP-1 agonists
            "GLP-1", "glucagon-like peptide-1", "glucagon-like peptide",
            # 약물명 (브랜드 + 성분명)
            "semaglutide", "tirzepatide", "liraglutide", "dulaglutide",
            "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Saxenda", "Victoza",
            # 비만 치료 일반
            "obesity treatment", "anti-obesity", "weight loss drug", "weight management",
            # 관련 용어
            "GIP/GLP-1", "incretin", "appetite suppression",
        ],
        "category": "대사질환",
        "why_hot": "비만→당뇨→심장→알츠하이머 적응증 확대, 노보/릴리 경쟁 격화"
    },

    # === 면역항암제 ===
    "CAR-T": {
        "keywords": ["CAR-T", "CAR T cell", "chimeric antigen receptor", "CAR T-cell",
                    "Kymriah", "Yescarta", "Tecartus", "Breyanzi", "Abecma", "Carvykti"],
        "category": "세포치료",
        "why_hot": "고형암 적용, 자가면역질환 확장"
    },
    "면역관문억제제": {
        "keywords": ["checkpoint inhibitor", "PD-1", "PD-L1", "CTLA-4",
                    "pembrolizumab", "Keytruda", "nivolumab", "Opdivo",
                    "atezolizumab", "Tecentriq", "durvalumab", "Imfinzi",
                    "immune checkpoint", "checkpoint blockade"],
        "category": "면역항암",
        "why_hot": "병용요법 확대, 내성 극복"
    },
    "이중항체": {
        "keywords": ["bispecific", "bispecific antibody", "T cell engager", "BiTE",
                    "bispecific T-cell", "dual-targeting", "Blincyto", "Tecvayli",
                    "Columvi", "Lunsumio", "Epkinly"],
        "category": "면역항암",
        "why_hot": "차세대 면역항암제, 혈액암에서 고형암으로 확장"
    },
    "ADC": {
        "keywords": ["ADC", "antibody-drug conjugate", "antibody drug conjugate",
                    "Enhertu", "Padcev", "Trodelvy", "Adcetris", "Kadcyla",
                    "payload", "linker technology"],
        "category": "항암제",
        "why_hot": "빅파마 M&A 러시, 차세대 항암제"
    },

    # === 유전자/세포치료 ===
    "CRISPR": {
        "keywords": ["CRISPR", "Cas9", "gene editing", "base editing", "prime editing",
                    "CRISPR-Cas", "Casgevy", "Exa-cel", "CRISPR Therapeutics", "Editas"],
        "category": "유전자편집",
        "why_hot": "최초 유전자치료제 카스게비 승인"
    },
    "유전자치료": {
        "keywords": ["gene therapy", "AAV", "adeno-associated virus", "viral vector",
                    "gene transfer", "Zolgensma", "Luxturna", "Hemgenix", "Elevidys"],
        "category": "플랫폼",
        "why_hot": "희귀질환 원샷 치료"
    },
    "세포치료": {
        "keywords": ["cell therapy", "stem cell therapy", "iPSC", "induced pluripotent",
                    "mesenchymal stem cell", "hematopoietic stem cell", "allogeneic cell"],
        "category": "플랫폼",
        "why_hot": "iPSC 기반 치료제 임상 확대"
    },
    "NK세포": {
        "keywords": ["NK cell", "natural killer cell", "NK cell therapy", "CAR-NK",
                    "NK cell immunotherapy", "off-the-shelf NK"],
        "category": "세포치료",
        "why_hot": "off-the-shelf 세포치료 대안"
    },

    # === RNA 플랫폼 ===
    "mRNA": {
        "keywords": ["mRNA", "messenger RNA", "mRNA vaccine", "mRNA therapeutics",
                    "mRNA-based", "mRNA cancer vaccine", "personalized mRNA",
                    "Moderna", "BioNTech"],
        "category": "플랫폼",
        "why_hot": "암백신, 희귀질환 확장"
    },
    "RNA치료제": {
        "keywords": ["siRNA", "antisense oligonucleotide", "ASO", "RNAi",
                    "oligonucleotide therapy", "Alnylam", "Ionis", "Arrowhead",
                    "Onpattro", "Leqvio", "Spinraza"],
        "category": "플랫폼",
        "why_hot": "희귀질환 치료제 승인 확대"
    },
    "LNP": {
        "keywords": ["LNP", "lipid nanoparticle", "nanoparticle delivery",
                    "lipid nanoparticles", "ionizable lipid", "mRNA delivery"],
        "category": "전달체",
        "why_hot": "mRNA 전달 핵심 기술"
    },

    # === 신경퇴행 ===
    "알츠하이머": {
        "keywords": ["Alzheimer", "amyloid", "tau protein", "amyloid-beta",
                    "lecanemab", "Leqembi", "donanemab", "Kisunla",
                    "aducanumab", "Aduhelm", "neurodegeneration", "dementia treatment"],
        "category": "신경퇴행",
        "why_hot": "레켐비/도나네맙 승인, 조기진단 바이오마커"
    },

    # === 감염병/COVID ===
    "COVID-19": {
        "keywords": ["COVID-19", "SARS-CoV-2", "coronavirus", "COVID vaccine",
                    "Paxlovid", "molnupiravir", "COVID treatment", "COVID booster"],
        "category": "감염병",
        "why_hot": "변이 대응, 백신/치료제 업데이트 지속"
    },
    "Long COVID": {
        "keywords": ["long COVID", "post-COVID", "PASC", "post-acute COVID",
                    "long-COVID", "post-COVID syndrome", "COVID sequelae"],
        "category": "감염병",
        "why_hot": "전세계 보건 이슈 지속"
    },

    # === AI/디지털 ===
    "AI 신약개발": {
        "keywords": ["AlphaFold", "AI drug discovery", "machine learning drug",
                    "deep learning drug", "artificial intelligence drug", "AI-driven drug",
                    "Insilico", "Recursion", "Exscientia", "Isomorphic Labs"],
        "category": "AI",
        "why_hot": "AlphaFold 3, 실용화 단계 진입"
    },

    # === 기타 주요 플랫폼 ===
    "Radiopharmaceuticals": {
        "keywords": ["radiopharmaceutical", "radioligand", "Pluvicto", "Lutathera",
                    "targeted radionuclide", "theranostics", "PSMA", "lutetium-177"],
        "category": "항암제",
        "why_hot": "전립선암/신경내분비종양 표적 방사선 치료"
    },
    "Microbiome": {
        "keywords": ["microbiome", "gut microbiota", "fecal microbiota", "FMT",
                    "microbiome therapy", "gut-brain axis", "Seres", "Ferring"],
        "category": "플랫폼",
        "why_hot": "장내 미생물 치료제 개발 가속화"
    },
    "Exosome": {
        "keywords": ["exosome", "extracellular vesicle", "exosome therapy",
                    "EV therapy", "exosome drug delivery"],
        "category": "플랫폼",
        "why_hot": "세포 유래 치료제 신기술"
    },
}


# 카테고리별 그룹화
TOPIC_CATEGORIES = {
    "대사질환": ["GLP-1/비만치료"],
    "세포치료": ["CAR-T", "세포치료", "NK세포"],
    "유전자편집": ["CRISPR"],
    "플랫폼": ["mRNA", "유전자치료", "LNP", "Microbiome", "RNA치료제", "Exosome"],
    "AI": ["AI 신약개발"],
    "면역항암": ["면역관문억제제", "이중항체", "ADC", "Radiopharmaceuticals"],
    "신경퇴행": ["알츠하이머"],
    "감염병": ["COVID-19", "Long COVID"],
}


# 급상승 키워드에서 제외할 일반적인 MeSH terms
EMERGING_BLACKLIST = {
    # 너무 일반적인 연구 방법론
    "disease models, animal", "animal experimentation",
    "clinical trials as topic", "randomized controlled trials as topic",
    # 일반적인 생물학 용어
    "mutation", "phenotype", "genotype", "biomarkers",
    # 일반적인 암 관련
    "carcinogenesis", "metastasis", "tumor burden",
}


def get_all_keywords() -> list:
    """Get all keywords from predefined topics."""
    all_keywords = []
    for topic_info in PREDEFINED_HOT_TOPICS.values():
        all_keywords.extend(topic_info["keywords"])
    return all_keywords


def is_in_predefined(keyword: str) -> bool:
    """Check if a keyword matches any predefined topic."""
    keyword_lower = keyword.lower()
    for topic_info in PREDEFINED_HOT_TOPICS.values():
        for kw in topic_info["keywords"]:
            if kw.lower() in keyword_lower or keyword_lower in kw.lower():
                return True
    return False


def find_matching_topic(text: str) -> list:
    """Find all matching predefined topics in text."""
    text_lower = text.lower()
    matches = []

    for topic_name, topic_info in PREDEFINED_HOT_TOPICS.items():
        for keyword in topic_info["keywords"]:
            if keyword.lower() in text_lower:
                matches.append(topic_name)
                break

    return matches


def get_topic_search_query(topic_name: str) -> str:
    """Get PubMed search query for a topic."""
    if topic_name not in PREDEFINED_HOT_TOPICS:
        return ""

    keywords = PREDEFINED_HOT_TOPICS[topic_name]["keywords"]
    # Use top 5 keywords for search
    query_parts = [f'"{kw}"[tiab]' for kw in keywords[:5]]
    return " OR ".join(query_parts)
