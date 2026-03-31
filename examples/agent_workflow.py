import logging

from nepal_gov_agent import GovRAG
from nepal_gov_agent.agent import GovAgent

logging.basicConfig(level=logging.INFO)

rag = GovRAG(corpus_dir="Data/")
agent = GovAgent(rag=rag, session_id="demo")

result = agent.run("What is the role of the National AI Centre?")
print(result.answer)
print("Confidence:", result.confidence)
print("Sources:", [(s["doc"], s["page"]) for s in result.sources[:3]])
