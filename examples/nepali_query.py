"""Example: single Nepali query against the Nepal government corpus."""

from nepal_gov_agent import GovRAG

if __name__ == "__main__":
    rag = GovRAG(corpus_dir="Data/")
    r = rag.ask("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
    print(r.answer[:1200])
    print("\nConfidence:", r.confidence)
