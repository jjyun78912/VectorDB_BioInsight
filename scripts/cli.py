#!/usr/bin/env python3
"""
VectorDB BioInsight - Bio Paper Vector Database

Main entry point for indexing and searching biomedical papers.

Usage:
    # Index papers
    python main.py index --domain pheochromocytoma --path ./data/papers

    # Search
    python main.py search --domain pheochromocytoma --query "genetic mutations"

    # RAG Q&A (requires Gemini API key)
    python main.py ask --domain pheochromocytoma --question "What are the genetic causes?"

    # Interactive chat mode
    python main.py chat --domain pheochromocytoma

    # Show stats
    python main.py stats --domain pheochromocytoma

    # Summarize a paper
    python main.py summarize --domain pheochromocytoma --title "Paper Title"
    python main.py summarize --pdf ./path/to/paper.pdf

    # Recommend related papers
    python main.py recommend --keyword "pheochromocytoma genetic mutations"
    python main.py recommend --domain pheochromocytoma --similar "Paper Title"
    python main.py recommend --pdf ./path/to/paper.pdf
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import create_indexer, BioPaperIndexer
from src.vector_store import create_vector_store


def cmd_index(args):
    """Index PDF papers into the vector database."""
    indexer = create_indexer(disease_domain=args.domain)

    if args.file:
        # Index single file
        result = indexer.index_pdf(args.file)
        if result.get("error"):
            print(f"Error: {result['error']}")
            return 1
    else:
        # Index directory
        path = Path(args.path) if args.path else Path("./data/papers")
        if not path.exists():
            print(f"Error: Path not found: {path}")
            return 1
        indexer.index_directory(path, recursive=args.recursive)

    return 0


def cmd_search(args):
    """Search the vector database."""
    vector_store = create_vector_store(disease_domain=args.domain)

    if vector_store.count == 0:
        print(f"No documents in collection. Run 'index' first.")
        return 1

    print(f"Searching: \"{args.query}\"")
    print(f"Domain: {args.domain}")
    print("-" * 60)

    if args.section:
        results = vector_store.search_by_section(args.query, args.section, top_k=args.top_k)
    else:
        results = vector_store.search(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return 0

    for i, result in enumerate(results, 1):
        score_bar = "█" * int(result.relevance_score / 5) + "░" * (20 - int(result.relevance_score / 5))
        print(f"\n[{i}] {score_bar} {result.relevance_score:.1f}%")
        print(f"    Paper: {result.metadata.get('paper_title', 'Unknown')[:60]}...")
        print(f"    Section: {result.metadata.get('section', 'Unknown')}")
        if result.metadata.get('parent_section'):
            print(f"    Parent: {result.metadata.get('parent_section')}")
        print(f"    Content: {result.content[:200]}...")

    return 0


def cmd_stats(args):
    """Show collection statistics."""
    vector_store = create_vector_store(disease_domain=args.domain)
    stats = vector_store.get_collection_stats()

    print(f"\n{'='*60}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Disease Domain: {stats['disease_domain']}")
    print(f"{'='*60}")
    print(f"Total Papers: {stats['total_papers']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Embedding Model: {stats['embedding_model']}")
    print(f"Embedding Dimension: {stats['embedding_dimension']}")

    print(f"\nChunks by Section:")
    for section, count in sorted(stats['chunks_by_section'].items(), key=lambda x: -x[1]):
        print(f"  - {section}: {count}")

    return 0


def cmd_list_papers(args):
    """List all indexed papers."""
    vector_store = create_vector_store(disease_domain=args.domain)
    papers = vector_store.get_all_papers()

    if not papers:
        print("No papers indexed yet.")
        return 0

    print(f"\nIndexed Papers ({len(papers)}):")
    print("-" * 60)
    for paper in papers:
        print(f"• {paper['title'][:70]}")
        if paper['doi']:
            print(f"  DOI: {paper['doi']}")
        if paper['year']:
            print(f"  Year: {paper['year']}")
        print()

    return 0


def cmd_delete(args):
    """Delete a paper from the index."""
    vector_store = create_vector_store(disease_domain=args.domain)

    if args.all:
        confirm = input("Delete ALL documents? This cannot be undone. [y/N]: ")
        if confirm.lower() == 'y':
            vector_store.reset()
            print("All documents deleted.")
        else:
            print("Cancelled.")
    elif args.title:
        deleted = vector_store.delete_paper(args.title)
        print(f"Deleted {deleted} chunks from paper: {args.title}")
    else:
        print("Specify --title or --all")
        return 1

    return 0


def cmd_ask(args):
    """Answer a question using RAG pipeline."""
    from src.rag_pipeline import create_rag_pipeline

    print(f"\n🔬 BioInsight RAG - {args.domain}")
    print("=" * 60)

    try:
        rag = create_rag_pipeline(
            disease_domain=args.domain,
            top_k=args.top_k
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if rag.vector_store.count == 0:
        print("No documents in collection. Run 'index' first.")
        return 1

    print(f"Question: {args.question}")
    print("-" * 60)

    response = rag.query(
        question=args.question,
        section=args.section
    )

    print(f"\n{response.format()}")

    return 0


def cmd_chat(args):
    """Interactive chat mode with RAG pipeline."""
    from src.rag_pipeline import create_rag_pipeline

    print(f"\n🔬 BioInsight RAG Chat - {args.domain}")
    print("=" * 60)
    print("Type your questions about the indexed papers.")
    print("Commands: /quit, /stats, /section <name>, /clear")
    print("=" * 60)

    try:
        rag = create_rag_pipeline(
            disease_domain=args.domain,
            top_k=args.top_k
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if rag.vector_store.count == 0:
        print("No documents in collection. Run 'index' first.")
        return 1

    stats = rag.vector_store.get_collection_stats()
    print(f"📚 Loaded {stats['total_papers']} papers ({stats['total_chunks']} chunks)")
    print()

    section_filter = None
    previous_context = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()

            if cmd[0] == "/quit":
                print("Goodbye!")
                break

            elif cmd[0] == "/stats":
                stats = rag.vector_store.get_collection_stats()
                print(f"\n📊 Collection Stats:")
                print(f"   Papers: {stats['total_papers']}")
                print(f"   Chunks: {stats['total_chunks']}")
                print(f"   Sections: {', '.join(stats['chunks_by_section'].keys())}")
                print()
                continue

            elif cmd[0] == "/section":
                if len(cmd) > 1:
                    section_filter = " ".join(cmd[1:])
                    print(f"📌 Section filter set to: {section_filter}")
                else:
                    section_filter = None
                    print("📌 Section filter cleared")
                continue

            elif cmd[0] == "/clear":
                previous_context = ""
                print("🔄 Conversation context cleared")
                continue

            else:
                print(f"Unknown command: {cmd[0]}")
                continue

        # RAG query
        print("\n🤔 Thinking...\n")

        response = rag.query(
            question=user_input,
            section=section_filter
        )

        print(f"Assistant: {response.format()}")

        # Save context for follow-up questions
        previous_context = response.answer[:500]

    return 0


def cmd_summarize(args):
    """Summarize a paper from vector DB or PDF file."""
    from src.summarizer import create_summarizer

    print(f"\n📝 BioInsight Paper Summarizer")
    print("=" * 70)

    try:
        if args.pdf:
            # Summarize from PDF file directly
            print(f"Summarizing PDF: {args.pdf}")
            print("-" * 70)

            summarizer = create_summarizer()
            summary = summarizer.summarize_pdf(
                args.pdf,
                include_sections=not args.brief
            )
        else:
            # Summarize from indexed paper
            if not args.domain:
                print("Error: --domain is required when summarizing indexed papers")
                return 1

            summarizer = create_summarizer(disease_domain=args.domain)

            if args.list:
                # List available papers
                papers = summarizer.list_indexed_papers()
                print(f"Available papers ({len(papers)}):")
                print("-" * 70)
                for i, paper in enumerate(papers, 1):
                    print(f"{i}. {paper['title'][:65]}...")
                    if paper.get('doi'):
                        print(f"   DOI: {paper['doi']}")
                return 0

            if not args.title:
                print("Error: --title is required. Use --list to see available papers.")
                return 1

            print(f"Summarizing: {args.title}")
            print(f"Domain: {args.domain}")
            print("-" * 70)

            summary = summarizer.summarize_from_vectordb(
                args.title,
                include_sections=not args.brief
            )

        print(f"\n{summary.format()}")

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_recommend(args):
    """Recommend related papers based on keywords or similarity."""
    from src.recommender import create_recommender

    print(f"\n🔍 BioInsight Paper Recommender")
    print("=" * 70)

    try:
        recommender = create_recommender(disease_domain=args.domain)

        if args.keyword:
            # Keyword-based recommendation
            print(f"Searching for: \"{args.keyword}\"")
            if args.min_year:
                print(f"Minimum year: {args.min_year}")
            print("-" * 70)
            print("Searching PubMed and fetching citation data...")

            papers = recommender.recommend_by_keyword(
                keywords=args.keyword,
                max_results=args.top_k,
                min_year=args.min_year
            )

        elif args.similar:
            # Similar paper recommendation (from indexed papers)
            if not args.domain:
                print("Error: --domain is required when using --similar")
                return 1

            print(f"Finding papers similar to: \"{args.similar}\"")
            print(f"Domain: {args.domain}")
            print("-" * 70)
            print("Calculating similarity and fetching recommendations...")

            papers = recommender.recommend_by_paper(
                paper_title=args.similar,
                max_results=args.top_k
            )

        elif args.pdf:
            # Similar paper recommendation (from PDF)
            print(f"Finding papers similar to PDF: {args.pdf}")
            print("-" * 70)
            print("Analyzing PDF and fetching recommendations...")

            papers = recommender.recommend_by_paper(
                pdf_path=args.pdf,
                max_results=args.top_k
            )

        else:
            print("Error: Provide --keyword, --similar, or --pdf")
            return 1

        if not papers:
            print("\nNo papers found. Try different keywords or check your internet connection.")
            return 0

        print(f"\n📚 Top {len(papers)} Recommended Papers")
        print("=" * 70)
        print("Score = Similarity (40%) + Recency (30%) + Impact (30%)")
        print("-" * 70)

        for i, paper in enumerate(papers, 1):
            print(paper.format(i))

        print("\n" + "=" * 70)
        print("💡 Tips:")
        print("   - Higher citation count = more established/validated research")
        print("   - Recent papers (2023-2025) may have fewer citations but newer findings")
        print("   - PMC links provide free full-text access")

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_trend(args):
    """Analyze research trends."""
    from src.trend_analyzer import create_trend_analyzer

    print(f"\n📊 Research Trend Analysis - {args.domain}")
    print("=" * 70)

    analyzer = create_trend_analyzer(disease_domain=args.domain)
    report = analyzer.analyze()

    print(report.format())
    return 0


def cmd_visualize(args):
    """Generate visualizations."""
    from src.visualizer import create_visualizer

    print(f"\n📊 Generating Visualizations - {args.domain}")
    print("=" * 70)

    visualizer = create_visualizer(disease_domain=args.domain)
    viz_data = visualizer.generate_all(output_dir=args.output)

    print(f"\n✅ Visualizations saved to: {args.output}")
    print("   Open index.html in a browser to view")
    return 0


def cmd_interpret(args):
    """Generate LLM interpretation report."""
    from src.interpreter import create_interpreter

    print(f"\n🔬 Research Interpretation - {args.domain}")
    print("=" * 70)

    try:
        interpreter = create_interpreter(disease_domain=args.domain)

        if args.compare:
            # Compare multiple papers
            titles = [t.strip() for t in args.compare.split(",")]
            print(f"Comparing {len(titles)} papers...")
            result = interpreter.compare_papers(titles)
            print(result)
        else:
            # Interpret single paper
            if not args.title:
                print("Error: --title is required")
                return 1

            print(f"Interpreting: {args.title}")
            print("-" * 70)
            report = interpreter.interpret_paper(args.title)
            print(report.format())

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_validate(args):
    """Validate research findings."""
    from src.validator import create_validator

    print(f"\n✅ Research Validation - {args.domain}")
    print("=" * 70)

    validator = create_validator(disease_domain=args.domain)

    if args.claim:
        print(f"Validating claim: \"{args.claim}\"")
        print("-" * 70)
        result = validator.validate_claim(args.claim)
        print(result.format())
    elif args.title:
        print(f"Validating paper consistency: {args.title}")
        print("-" * 70)
        result = validator.validate_paper_consistency(args.title)
        print(result.format())
    elif args.cross:
        # Cross-validate all papers
        from src.vector_store import create_vector_store
        vs = create_vector_store(disease_domain=args.domain)
        papers = vs.get_all_papers()
        titles = [p["title"] for p in papers]

        print(f"Cross-validating {len(titles)} papers...")
        print("-" * 70)
        results = validator.cross_validate_papers(titles)
        for result in results:
            print(f"\n📄 {result.item_validated[:50]}...")
            print(f"   Confidence: {result.overall_confidence:.1f}%")
    else:
        print("Error: Specify --claim, --title, or --cross")
        return 1

    return 0


def cmd_store(args):
    """Manage research asset storage."""
    from src.research_store import create_research_store

    store = create_research_store(disease_domain=args.domain)

    if args.action == "stats":
        stats = store.get_statistics()
        print(f"\n📦 Research Store Statistics - {args.domain}")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.action == "export":
        output = store.export_all()
        print(f"✅ Exported to: {output}")

    elif args.action == "history":
        logs = store.get_qa_history()
        print(f"\n📝 Q&A History ({len(logs)} entries)")
        print("=" * 60)
        for log in logs[:20]:
            print(f"\n[{log.timestamp}]")
            print(f"Q: {log.question[:80]}...")
            print(f"A: {log.answer[:100]}...")

    elif args.action == "papers":
        papers = store.list_papers()
        print(f"\n📚 Stored Papers ({len(papers)})")
        print("=" * 60)
        for p in papers:
            flags = []
            if p["has_summary"]:
                flags.append("📝")
            if p["has_interpretation"]:
                flags.append("🔬")
            if p["notes_count"] > 0:
                flags.append(f"📌{p['notes_count']}")
            print(f"  {' '.join(flags)} {p['title'][:50]}...")

    else:
        print("Unknown action. Use: stats, export, history, papers")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB BioInsight - Bio Paper Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index PDF papers")
    index_parser.add_argument("--domain", "-d", required=True, help="Disease domain (e.g., pheochromocytoma)")
    index_parser.add_argument("--path", "-p", help="Directory containing PDFs")
    index_parser.add_argument("--file", "-f", help="Single PDF file to index")
    index_parser.add_argument("--recursive", "-r", action="store_true", default=True, help="Search subdirectories")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the vector database")
    search_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    search_parser.add_argument("--query", "-q", required=True, help="Search query")
    search_parser.add_argument("--section", "-s", help="Filter by section")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection statistics")
    stats_parser.add_argument("--domain", "-d", required=True, help="Disease domain")

    # List command
    list_parser = subparsers.add_parser("list", help="List indexed papers")
    list_parser.add_argument("--domain", "-d", required=True, help="Disease domain")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete papers from index")
    delete_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    delete_parser.add_argument("--title", "-t", help="Paper title to delete")
    delete_parser.add_argument("--all", action="store_true", help="Delete all documents")

    # Ask command (RAG Q&A)
    ask_parser = subparsers.add_parser("ask", help="Ask a question using RAG (requires Gemini API)")
    ask_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    ask_parser.add_argument("--question", "-q", required=True, help="Your question")
    ask_parser.add_argument("--section", "-s", help="Filter by section (e.g., Methods)")
    ask_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of sources to retrieve")

    # Chat command (Interactive RAG)
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with RAG")
    chat_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    chat_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of sources to retrieve")

    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Summarize a research paper")
    summarize_parser.add_argument("--domain", "-d", help="Disease domain (for indexed papers)")
    summarize_parser.add_argument("--title", "-t", help="Paper title to summarize")
    summarize_parser.add_argument("--pdf", "-p", help="PDF file path to summarize directly")
    summarize_parser.add_argument("--list", "-l", action="store_true", help="List available papers")
    summarize_parser.add_argument("--brief", "-b", action="store_true", help="Brief summary (no section details)")

    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Recommend related papers")
    recommend_parser.add_argument("--keyword", "-k", help="Search by keywords")
    recommend_parser.add_argument("--similar", "-s", help="Find papers similar to indexed paper title")
    recommend_parser.add_argument("--pdf", "-p", help="Find papers similar to PDF file")
    recommend_parser.add_argument("--domain", "-d", help="Disease domain (for --similar option)")
    recommend_parser.add_argument("--top-k", "-n", type=int, default=10, help="Number of recommendations")
    recommend_parser.add_argument("--min-year", "-y", type=int, help="Minimum publication year")

    # Trend analysis command
    trend_parser = subparsers.add_parser("trend", help="Analyze research trends")
    trend_parser.add_argument("--domain", "-d", required=True, help="Disease domain")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    viz_parser.add_argument("--output", "-o", default="./output/viz", help="Output directory")

    # Interpret command
    interpret_parser = subparsers.add_parser("interpret", help="LLM-based paper interpretation")
    interpret_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    interpret_parser.add_argument("--title", "-t", help="Paper title to interpret")
    interpret_parser.add_argument("--compare", "-c", help="Compare papers (comma-separated titles)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate research findings")
    validate_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    validate_parser.add_argument("--claim", help="Validate a specific claim")
    validate_parser.add_argument("--title", "-t", help="Validate paper consistency")
    validate_parser.add_argument("--cross", action="store_true", help="Cross-validate all papers")

    # Store command
    store_parser = subparsers.add_parser("store", help="Manage research asset storage")
    store_parser.add_argument("--domain", "-d", required=True, help="Disease domain")
    store_parser.add_argument("action", choices=["stats", "export", "history", "papers"], help="Action to perform")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "stats": cmd_stats,
        "list": cmd_list_papers,
        "delete": cmd_delete,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "summarize": cmd_summarize,
        "recommend": cmd_recommend,
        "trend": cmd_trend,
        "visualize": cmd_visualize,
        "interpret": cmd_interpret,
        "validate": cmd_validate,
        "store": cmd_store,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
