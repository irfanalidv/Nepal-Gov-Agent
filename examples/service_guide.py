import logging

from nepal_gov_agent import GovRAG
from nepal_gov_agent.agent import GovAgent

logging.basicConfig(level=logging.INFO)

rag = GovRAG(corpus_dir="Data/")
agent = GovAgent(rag=rag, session_id="service_guide_demo")

result = agent.run(
    "How do I apply for a citizenship certificate renewal in Nepal?",
    workflow="service_guide",
)
print(result.answer)
print("\nSteps run: %d" % len(result.steps))
print("Sources: %d unique blocks" % len(result.sources))
