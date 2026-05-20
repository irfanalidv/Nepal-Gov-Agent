# Synthetic Evaluation on the Nepal Government Corpus

*Written for the v0.4.0 release of [nepal-gov-agent](https://pypi.org/project/nepal-gov-agent/). Last updated alongside the release notes.*

## TL;DR

Nepal GovAgent v0.4.0 ships a synthetic evaluation harness. It uses GPT-4.1-mini to generate questions from every page of the seed PDFs, validates that the expected keywords appear in the source text, and runs them through the retriever to compute Recall@k and a few language-stratified metrics.

**Treat the numbers as a regression signal, not as evidence of capability.** The eval set was written by a model, not by a Nepali-speaking civil servant or lawyer. Anyone citing these numbers as a benchmark in the conventional sense is misrepresenting them. The report header and the dataset README both carry that disclaimer.

This post explains why we still bothered to build it, what it does and doesn't tell us, and what a real evaluation would look like.

## Why synthetic eval at all

The original benchmark in this repo had seven hand-curated questions. That's enough to catch egregious regressions (retrieval is completely broken, the wrong PDF is loaded, the embedding model isn't applied) but not enough to compare two configurations. If you tune `w_bm25` from 0.5 to 0.6 and Recall@3 moves from 0.857 to 0.714, did anything actually change? On seven questions, the answer is "you can't tell."

The choices were:

1. Hire native Nepali-speaking reviewers to write a few hundred questions. Right answer. Out of scope for a maintenance-mode reference implementation.
2. Use one of the public Nepali QA datasets. Nepali QA datasets exist but none of them are over policy and legal documents specifically.
3. Generate questions with an LLM, validate them mechanically, ship them with a loud disclaimer.

Option 3 is what we did. It is honest about what it is. It is not a substitute for option 1.

## How the harness works

For each PDF in the seed corpus:

1. Iterate over every page.
2. Skip pages with fewer than 40 meaningful words (covers, blank pages, tables of contents).
3. Send the page text to GPT-4.1-mini with a strict prompt: generate exactly two questions answerable from this passage, one in English and one in Nepali, plus 3–5 short keywords that must appear verbatim in the passage.
4. For each returned pair, check every keyword against the source page after NFC normalisation and lowercasing. If any keyword is missing, drop the pair.
5. Cap per-document count so one long PDF doesn't dominate the eval.
6. Persist surviving pairs as JSONL with provenance — source page, source excerpt, generator model.

Generation is idempotent. Pages already covered in the output file are skipped on re-run.

The validation step is the load-bearing part of the honesty story. We do not trust the model's claim that a keyword is "in the passage" — we check, character by character, after Unicode normalisation. Pairs that fail this check never reach the JSONL file. In practice, GPT-4.1-mini gets this right roughly 90% of the time on English text and somewhat less reliably on the Nepali side, where it occasionally invents Devanagari spelling variants that don't quite match the source.

## What the numbers do tell us

When run against the seed corpus on default settings, the synthetic eval produces stable Recall@k numbers that move in predictable ways when you change the underlying configuration:

- **Embedding model swap**: replacing `multilingual-e5-small` with a larger model shifts Recall@3 by a few points and shifts the Nepali-vs-English gap differently than the English-only gap.
- **BM25/vector weight tuning**: small movements in `w_bm25` produce small, monotonic movements in keyword hit rate. Large movements produce non-monotonic effects on Nepali queries specifically — which is interesting and probably worth real investigation.
- **Chunking strategy changes upstream in RAGNav**: visible in Recall@1 specifically. Doc hit rate stays high regardless.

These are all *signals*. They tell us whether a change broke something or moved a needle. They do not tell us whether the system is good.

## First-run findings (v0.4.0)

When we ran the harness for the first time on the seed corpus, two numbers stood out:

- **English Recall@3: 0.707**
- **Nepali Recall@3: 0.429**

The same retrieval pipeline, the same documents, the same scoring code — only the query language differs. That ~28-point gap is the single most useful thing this eval has surfaced, and it's worth being explicit about how to read it.

**What the gap most likely is.** The retrieval architecture (hybrid BM25 + dense vector, RAGNav's chunking, ragfallback's query rewriting) is shared across both languages and works fine on the English side. The component that is *not* shared in any meaningful sense is the embedding model — `multilingual-e5-small` is a 118M-parameter encoder trained on a broad multilingual mix in which Nepali (and Devanagari-script policy register specifically) is a small slice. The lexical BM25 component also gives Nepali less leverage: government policy PDFs frequently mix Devanagari with English technical terms, and a Nepali query against an English passage cannot win on lexical overlap.

The first-order interpretation is therefore: the retrieval *architecture* is not the bottleneck for Nepali; the *embedding layer* is. A larger multilingual encoder (e.g. `multilingual-e5-large`, `BAAI/bge-m3`) or a Nepali-tuned encoder would probably close a large fraction of this gap without any architectural changes. We did not try this in v0.4.0 because swapping the model invalidates the embedding cache and triples generation cost, and the goal of this release was the harness, not the optimization.

**What the gap might not be — the test-as-artefact problem.** Before anyone reads "Nepali retrieval is broken," the honesty disclaimer from the rest of this document has to apply with extra weight here. The Nepali questions in the synthetic eval were generated by GPT-4.1-mini, not written by a native Nepali speaker. Some fraction of the 0.429 figure may not be retrieval failure at all — it may be that the generator produced awkward Nepali phrasings, register mismatches between formal Devanagari policy language and conversational query language, or Devanagari spellings that drift from official government usage. The retriever then fails to find a match because the *question* doesn't sound like the way a real user would ask, not because the *retrieval* is bad.

We don't currently know how to apportion the gap between "model weakness" and "test-as-artefact." That decomposition requires native-speaker review of the Nepali side of the eval set, which is the highest-leverage piece of work left on this project. If a fluent Nepali speaker flagged the unidiomatic questions and we re-ran the benchmark on just the idiomatic ones, the Nepali Recall@3 would move — possibly substantially. We don't know in which direction it would move further, only that the current 0.429 is a *combined* failure mode of model and eval set, and that combination cannot be unmixed without the human review.

**Author eyeball pass on the Nepali set.** Of the 21 Nepali pairs in `eval_data/synthetic_qa_v1.jsonl`, roughly half read as plausible formal-policy questions; the rest mix English technical terms, formal/colloquial register inconsistencies, and OCR artefacts in the expected-keywords field inherited from the source PDFs (e.g. `प्रतितिति`, `धनयमावली`, `धसफाररस`, `कम्प्युिर अपरेिर`). The eval set is also corpus-skewed: 10 of 21 Nepali pairs come from the Legal Maxims PDF, and zero come from the Constitution or the National AI Policy — both of which carry primarily English text where the Nepali side of the eval has nothing to retrieve against. So the 0.429 figure is measuring a particular subset of the corpus, not Nepali retrieval across the whole.

The OCR-artefacts-in-keywords finding is worth naming because it's a limitation of the validation logic, not just of the eval set. The "keywords must appear verbatim in the source passage" rule was designed to prevent the generator from hallucinating answers — and it does that well. But it silently passes pairs where *both* the question keywords and the source page contain the same OCR corruption, since the byte-level match still succeeds. A more careful harness would normalize OCR-likely substitutions (Devanagari character confusions in particular) before the validation step, or flag pairs whose keywords don't appear in any Unicode-normalized dictionary. Out of scope for v0.4.0; logged as a known limitation.

**What this means for the rest of the disclaimer.** Nothing changes. The 0.707 English figure is also a combined number — model + eval + retrieval interacting. It just looks better because the eval-set noise on the English side is smaller (the generator produces native-fluency English questions). The right way to read both numbers is: *useful as a regression signal between two runs of this code against this corpus; not useful as a benchmark statement about retrieval quality in any deployable sense.*

If you fork this project or adapt it for another low-resource language corpus, expect the same shape: the cross-language gap you measure with synthetic eval will be a real signal mixed with a real artefact, and you cannot separate them without bringing in a native speaker.

## What the numbers do not tell us

Four specific things:

1. **They do not tell us if the questions are sensible.** Validation only checks that keywords appear in the source. It does not check that the question is a reasonable thing someone might ask, or that the keywords actually answer it. A page about ministry organisational structure might generate a question like "What is the role of the Joint Secretary?" with keywords like "joint secretary" — and the page might mention the title in passing without explaining the role at all. The pair would pass validation and the retriever would happily return the page. Neither party has actually demonstrated anything useful.

2. **They do not tell us if the Nepali questions are well-formed Nepali.** They're LLM-translated. A native speaker would catch awkward phrasings, register mismatches between formal policy language and conversational query language, and Devanagari spellings that drift from official government usage. The eval can't see any of that.

3. **They do not tell us about multi-hop reasoning, ambiguity, or exceptions.** Every question is, by construction, answerable from a single passage. Real queries against policy and legal corpora often require synthesising across documents, reasoning about date precedence, or recognising that a question is asking about an exception rather than the general rule. The synthetic eval has zero coverage of any of this.

4. **The eval and the system share a substrate.** The generator read the same text the retriever reads. Lexical overlap is easy to achieve when both ends use embeddings trained on similar distributions. Numbers will look better here than they should. A proper benchmark would have questions written by someone who never saw the corpus directly, asking what they actually wanted to know.

## What a real evaluation looks like

In rough order of cost:

1. **Native-speaker review of a sample.** Pick 50 random pairs from the synthetic set, hand them to a fluent Nepali speaker, ask them to flag awkward phrasings and questions whose expected keywords don't actually answer the question. The flag rate is itself the most useful number we can publish from the synthetic eval.

2. **Adversarial questions from a domain expert.** A lawyer or civil servant who works with these documents writes 50–100 questions specifically targeting the kinds of queries the system would face in production — dates, amendments, eligibility edge cases, exceptions. Cost: a few days of expert time.

3. **Held-out user queries.** If the system is ever deployed (a real pilot, not the demo), capture the questions users actually ask. Use them to build the next benchmark generation. This is the only evaluation that matters in the end.

4. **Inter-annotator agreement.** Have two domain experts independently answer the same 50 questions from the corpus. The disagreement rate is the ceiling on any automated metric. If humans disagree 20% of the time, no system can score above 80% on those questions without overfitting to one annotator's view.

The synthetic eval is a starting point for #1. It is not a substitute for any of #2 through #4.

## How to read the numbers in our reports

If you see a synthetic-eval number from this project, the report will carry this header:

```
Eval set: SYNTHETIC eval — LLM-generated, NOT human-validated
Generator: gpt-4.1-mini
```

And this footer:

```
⚠️  These numbers come from an LLM-generated eval set. Use them
    as a regression signal, not as ground truth. See
    eval_data/README.md and docs/synthetic-eval.md.
```

If you see the numbers cited somewhere without that context, that's the citer's error, not the project's. The disclaimer is structurally attached to the numbers wherever they're rendered.

## What's next

For this project, not much — it's in maintenance. v0.4.0 was the polish pass. If a Nepali-speaking contributor wants to volunteer to review the synthetic set and curate a smaller human-validated subset, open an issue.

For RAG over policy and legal corpora more broadly, the more interesting work is on the agentic side — multi-hop questions, structured extraction of dates and citations, cross-document reasoning. None of that fits in a smoke-test benchmark; it needs the kind of evaluation infrastructure described above, which is its own multi-month project. Out of scope here, in scope for whoever picks the next step up.

---

*Code: [github.com/irfanalidv/Nepal-Gov-Agent](https://github.com/irfanalidv/Nepal-Gov-Agent). Demo: [nepalgov.datacortex.in](https://nepalgov.datacortex.in). Author: [Irfan Ali](https://datacortex.in), DataCortex IQ.*
