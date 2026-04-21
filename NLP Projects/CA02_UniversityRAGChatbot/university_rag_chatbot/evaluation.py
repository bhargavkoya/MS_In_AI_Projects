"""
=============================================================
 evaluation.py  –  RAG Evaluation Module
 University Knowledge Base Chatbot
 MSc AI | NLP Module (B9AI006) | Dublin Business School
=============================================================

Evaluation Strategy
───────────────────
Two complementary evaluation approaches are combined:

  1. Lightweight Proxy Metrics (no ground-truth required)
     ─ Keyword Coverage Score  : fraction of expected keywords in the answer
     ─ Source Citation Rate    : whether at least one grounded source was returned
     ─ Response Time           : RAG pipeline latency
     ─ Answer Conciseness      : word count of generated answers

  2. RAGAS Metrics (Retrieval-Augmented Generation Assessment)
     ─ Faithfulness            : is the answer grounded in the retrieved context?
     ─ Answer Relevancy        : does the answer actually address the question?
     ─ Context Precision       : are the retrieved chunks ranked well? (requires ground_truth)
     ─ Context Recall          : were all necessary chunks retrieved?  (requires ground_truth)

     RAGAS reference: Es et al. (2023). arXiv:2309.15217
     All RAGAS metrics are computed without human-annotated gold answers
     (faithfulness + answer_relevancy) or with lightweight reference answers
     (context_precision + context_recall, when ground_truth is present in test_set.json).

Usage
─────
    python evaluation.py
    (ensure ANTHROPIC_API_KEY is set in .env and vectorstore/ exists)
"""

import json
import time
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuestionResult:
    """Stores all metrics for a single evaluation question."""
    question:           str
    category:           str
    expected_keywords:  List[str]

    # Answer
    rag_answer:         str

    # Keyword Coverage (0.0 – 1.0)
    rag_keyword_score:  float

    # Timing
    rag_response_time:  float

    # Source transparency
    rag_source_count:   int
    rag_has_source:     bool

    # Answer length (word count)
    rag_word_count:     int

    # Raw context chunks (page_content strings) – passed to RAGAS
    rag_contexts:       List[str] = field(default_factory=list)

    # Optional reference answer – enables RAGAS context_precision + context_recall
    ground_truth:       str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Default Test Set
# ══════════════════════════════════════════════════════════════════════════════

def _load_default_test_set() -> List[Dict]:
    """Load test questions from test_set.json if it exists, else return empty list."""
    test_set_path = Path("test_set.json")
    if test_set_path.exists():
        with open(test_set_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {
            "question": "What is the submission deadline for the NLP CA2 assignment?",
            "expected_keywords": ["deadline", "week 12", "submission", "sunday", "23:55"],
            "category": "NLP – Submission Policy",
        },
        {
            "question": "How many marks is the group presentation worth?",
            "expected_keywords": ["30", "marks", "presentation", "group", "task 1"],
            "category": "NLP – Assessment Details",
        },
    ]


DEFAULT_TEST_SET = _load_default_test_set()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator Class
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Systematic evaluation of the RAG pipeline combining keyword-based proxy
    metrics and RAGAS automated evaluation.

    Methods
    -------
    run_evaluation()        – Execute full evaluation and print proxy metrics
    run_ragas_evaluation()  – Run RAGAS metrics on the collected results
    print_summary()         – Print aggregate keyword/latency metrics table
    export_results()        – Save all results (including RAGAS scores) to JSON
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.results: List[QuestionResult] = []
        self.ragas_scores: Optional[Dict] = None        # populated by run_ragas_evaluation()
        self.ragas_per_question: List[Dict] = []        # per-question RAGAS breakdown

    # ── Proxy Metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def keyword_coverage(answer: str, keywords: List[str]) -> float:
        """
        Keyword Coverage Score
        ──────────────────────
        Counts how many expected keywords (case-insensitive) appear in the
        generated answer. Expressed as a fraction in [0, 1].

        Limitation: penalises paraphrased correct answers.
        RAGAS Answer Relevancy provides a semantic alternative.
        """
        if not keywords:
            return 0.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return round(hits / len(keywords), 3)

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    # ── Main Evaluation Loop ──────────────────────────────────────────────────

    def run_evaluation(
        self,
        test_set: List[Dict] = None,
        verbose: bool = True,
    ) -> List[QuestionResult]:
        """
        Run proxy-metric evaluation on every question in test_set.

        For each question:
          - Call answer_with_rag()
          - Capture answer, source contexts, response time
          - Compute keyword coverage, word count, citation presence
        """
        if test_set is None:
            test_set = DEFAULT_TEST_SET

        print("\n" + "=" * 70)
        print("  EVALUATION:  RAG Pipeline  (Keyword + Latency Metrics)")
        print("=" * 70)
        print(f"  Test set size : {len(test_set)} questions")
        print(f"  Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        results = []

        for idx, item in enumerate(test_set, 1):
            question = item["question"]
            keywords = item.get("expected_keywords", [])
            category = item.get("category", "General")
            ground_truth = item.get("ground_truth", "")

            if verbose:
                print(f"\n[{idx:02d}/{len(test_set)}]  {category}")
                print(f"  Q: {question}")
                print("  " + "-" * 65)

            # ── Get RAG answer ─────────────────────────────────────────────
            rag_res = self.pipeline.answer_with_rag(question)

            # ── Capture raw context chunks for RAGAS ───────────────────────
            rag_contexts = [doc.page_content for doc in rag_res["sources"]]

            # ── Compute proxy metrics ──────────────────────────────────────
            rag_score = self.keyword_coverage(rag_res["answer"], keywords)
            rag_wc    = self.word_count(rag_res["answer"])
            has_source = rag_res["num_sources"] > 0

            result = QuestionResult(
                question          = question,
                category          = category,
                expected_keywords = keywords,
                rag_answer        = rag_res["answer"],
                rag_keyword_score = rag_score,
                rag_response_time = rag_res["response_time"],
                rag_source_count  = rag_res["num_sources"],
                rag_has_source    = has_source,
                rag_word_count    = rag_wc,
                rag_contexts      = rag_contexts,
                ground_truth      = ground_truth,
            )
            results.append(result)

            if verbose:
                print(f"  Keyword score : {rag_score:.0%}")
                print(f"  Response time : {rag_res['response_time']:.2f}s")
                print(f"  Sources       : {rag_res['num_sources']}  |  Words: {rag_wc}")
                print(f"\n  Answer (snippet):")
                snippet = rag_res["answer"][:200]
                print(f"    {snippet} ...")

        self.results = results
        self.print_summary(results)
        return results

    # ── Summary Table ─────────────────────────────────────────────────────────

    def print_summary(self, results: List[QuestionResult]) -> None:
        """Print aggregate proxy evaluation metrics."""
        n = len(results)
        if n == 0:
            print("No results to summarise.")
            return

        avg_kw    = sum(r.rag_keyword_score  for r in results) / n
        avg_time  = sum(r.rag_response_time  for r in results) / n
        avg_wc    = sum(r.rag_word_count     for r in results) / n
        cite_rate = sum(1 for r in results if r.rag_has_source) / n
        high      = sum(1 for r in results if r.rag_keyword_score >= 0.6)
        low       = sum(1 for r in results if r.rag_keyword_score < 0.4)

        print("\n" + "=" * 70)
        print("  PROXY METRICS SUMMARY  –  RAG Pipeline")
        print("=" * 70)
        print(f"  {'Total Questions':<35} {n}")
        print(f"  {'High Coverage (>=60%)':<35} {high}/{n}  ({high/n:.0%})")
        print(f"  {'Low Coverage (<40%)':<35} {low}/{n}  ({low/n:.0%})")
        print("-" * 70)
        print(f"  {'Avg Keyword Coverage':<35} {avg_kw:.1%}")
        print("-" * 70)
        print(f"  {'Avg Response Time':<35} {avg_time:.2f}s")
        print("-" * 70)
        print(f"  {'Avg Answer Length (words)':<35} {avg_wc:.0f}")
        print("-" * 70)
        print(f"  {'Source Citation Rate':<35} {cite_rate:.1%}")
        print("=" * 70)

        if avg_kw >= 0.6:
            print(f"\n  Good keyword coverage – {avg_kw:.1%} avg.\n")
        elif avg_kw >= 0.4:
            print(f"\n  Moderate keyword coverage ({avg_kw:.1%}). Consider tuning chunk size or top-K.\n")
        else:
            print(f"\n  Low keyword coverage ({avg_kw:.1%}). Check PDF content and chunk settings.\n")

    # ── RAGAS Evaluation ──────────────────────────────────────────────────────

    def run_ragas_evaluation(self) -> Optional[Dict]:
        """
        Run RAGAS automated evaluation on collected results.

        Metrics used
        ────────────
        Always computed (no ground_truth required):
          - Faithfulness      : fraction of answer claims supported by retrieved context.
                                High score = answer is grounded; low = hallucination risk.
          - Answer Relevancy  : semantic similarity between answer and question.
                                High score = answer addresses the question directly.

        Computed when ground_truth is present in test_set.json entries:
          - Context Precision : are the most relevant chunks ranked highest?
                                High score = retriever surfaces best chunks first.
          - Context Recall    : does the retrieved context cover the reference answer?
                                High score = retriever captures all necessary information.

        RAGAS uses the same Anthropic LLM and HuggingFace embeddings as the pipeline,
        so no additional API keys are required beyond ANTHROPIC_API_KEY.

        Returns
        -------
        dict of aggregate RAGAS scores, or None if RAGAS is not installed.
        """
        if not self.results:
            print("No results. Run run_evaluation() first.")
            return None

        # ── Import RAGAS ───────────────────────────────────────────────────────
        try:
            from ragas import evaluate, EvaluationDataset, SingleTurnSample
            from ragas.metrics import Faithfulness, AnswerRelevancy
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except ImportError:
            print(
                "\n  RAGAS not installed.\n"
                "  Run:  pip install 'ragas>=0.2.0' datasets\n"
                "  Then re-run evaluation.\n"
            )
            return None

        # ── Determine which metrics are available ──────────────────────────────
        has_ground_truth = any(r.ground_truth.strip() for r in self.results)

        metrics = [Faithfulness(), AnswerRelevancy()]

        if has_ground_truth:
            try:
                from ragas.metrics import ContextPrecision, ContextRecall
                metrics += [ContextPrecision(), ContextRecall()]
            except ImportError:
                pass    # older RAGAS build – skip context metrics

        metric_names = [m.name for m in metrics]

        # ── Configure LLM and embeddings ───────────────────────────────────────
        # Reuse the same models already loaded by the pipeline to avoid
        # downloading additional weights.
        from langchain_anthropic import ChatAnthropic
        from langchain_community.embeddings import HuggingFaceEmbeddings

        ragas_llm = LangchainLLMWrapper(
            ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.0)
        )
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        # ── Build RAGAS dataset ────────────────────────────────────────────────
        # SingleTurnSample maps to one Q&A exchange.
        # retrieved_contexts must be a non-empty list of strings.
        samples = []
        for r in self.results:
            contexts = r.rag_contexts if r.rag_contexts else ["No context retrieved."]
            sample = SingleTurnSample(
                user_input         = r.question,
                response           = r.rag_answer,
                retrieved_contexts = contexts,
                reference          = r.ground_truth if (has_ground_truth and r.ground_truth.strip()) else None,
            )
            samples.append(sample)

        dataset = EvaluationDataset(samples=samples)

        # ── Run RAGAS ──────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  RAGAS EVALUATION  –  Automated RAG Quality Metrics")
        print("=" * 70)
        print(f"  Metrics       : {', '.join(metric_names)}")
        print(f"  Questions     : {len(self.results)}")
        gt_count = sum(1 for r in self.results if r.ground_truth.strip())
        print(f"  Ground Truth  : {gt_count}/{len(self.results)} questions have reference answers")
        if not has_ground_truth:
            print("  Note          : Add 'ground_truth' to test_set.json entries to enable")
            print("                  context_precision and context_recall metrics.")
        print("\n  Running RAGAS (each question calls the LLM – may take a few minutes) ...")
        print("-" * 70)

        scores = evaluate(
            dataset    = dataset,
            metrics    = metrics,
            llm        = ragas_llm,
            embeddings = ragas_emb,
        )

        # ── Store aggregate scores ─────────────────────────────────────────────
        # EvaluationResult supports scores[metric_name] via __getitem__,
        # not .items() – pull each metric by name explicitly.
        self.ragas_scores = {}
        for metric in metrics:
            try:
                self.ragas_scores[metric.name] = float(scores[metric.name])
            except (KeyError, TypeError):
                pass

        # ── Store per-question scores ──────────────────────────────────────────
        try:
            df = scores.to_pandas()
            self.ragas_per_question = df.to_dict(orient="records")
        except Exception:
            self.ragas_per_question = []

        # ── Print results ──────────────────────────────────────────────────────
        faith = self.ragas_scores.get("faithfulness", None)
        relev = self.ragas_scores.get("answer_relevancy", None)
        prec  = self.ragas_scores.get("context_precision", None)
        rec   = self.ragas_scores.get("context_recall", None)

        print(f"\n  Faithfulness")
        print(f"    Score  : {faith:.3f}" if faith is not None else "    Score  : N/A")
        print(f"    Meaning: Fraction of answer claims traceable to retrieved context.")
        print(f"             A score of 1.0 means every claim is grounded; 0.0 = full hallucination.")

        print(f"\n  Answer Relevancy")
        print(f"    Score  : {relev:.3f}" if relev is not None else "    Score  : N/A")
        print(f"    Meaning: Semantic alignment between the answer and the question.")
        print(f"             A score near 1.0 means the answer directly addresses the question.")

        if prec is not None:
            print(f"\n  Context Precision")
            print(f"    Score  : {prec:.3f}")
            print(f"    Meaning: Whether the most relevant chunks are ranked first.")
            print(f"             Low score suggests reranking could improve retrieval order.")

        if rec is not None:
            print(f"\n  Context Recall")
            print(f"    Score  : {rec:.3f}")
            print(f"    Meaning: Whether the retrieved context covers the reference answer.")
            print(f"             Low score suggests important chunks are not being retrieved.")

        # ── Overall verdict ────────────────────────────────────────────────────
        scored_values = [v for v in [faith, relev, prec, rec] if v is not None]
        overall = sum(scored_values) / len(scored_values) if scored_values else 0.0

        print("\n" + "-" * 70)
        print(f"  Overall RAGAS Score   : {overall:.3f}  (mean of computed metrics)")
        print("=" * 70)

        if overall >= 0.75:
            print(f"\n  Good RAGAS quality ({overall:.3f}). Pipeline is retrieving and grounding well.\n")
        elif overall >= 0.5:
            print(f"\n  Moderate RAGAS quality ({overall:.3f}). Consider hybrid search or reranking.\n")
        else:
            print(f"\n  Low RAGAS quality ({overall:.3f}). Review chunk size, top-K, and corpus coverage.\n")

        return self.ragas_scores

    # ── Export ────────────────────────────────────────────────────────────────

    def export_results(self, output_path: str = "evaluation_results.json") -> None:
        """Save all proxy and RAGAS results to a JSON file for further analysis."""
        n = len(self.results)
        export = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": n,
                # Proxy metric aggregates
                "avg_rag_keyword_score":
                    sum(r.rag_keyword_score for r in self.results) / n if n else 0,
                "avg_rag_response_time":
                    sum(r.rag_response_time for r in self.results) / n if n else 0,
                "source_citation_rate":
                    sum(1 for r in self.results if r.rag_has_source) / n if n else 0,
                # RAGAS aggregate scores (empty dict if RAGAS was not run)
                "ragas_scores": self.ragas_scores or {},
            },
            # Per-question proxy results (includes rag_contexts and ground_truth)
            "results": [asdict(r) for r in self.results],
            # Per-question RAGAS breakdown (populated after run_ragas_evaluation)
            "ragas_per_question": self.ragas_per_question,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"\n  Results exported to '{output_path}'")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to a .env file:\n  ANTHROPIC_API_KEY=sk-ant-..."
        )

    # ── 1. Initialise pipeline ────────────────────────────────────────────────
    pipeline = RAGPipeline(
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
        temperature=0.0,
    )

    # ── 2. Load pre-built vector store ────────────────────────────────────────
    vectorstore_path = "vectorstore"
    if not Path(vectorstore_path).exists():
        raise FileNotFoundError(
            f"No vector store found at '{vectorstore_path}'.\n"
            "Run:  python ingest.py --pdf_dir data\n"
            "before running evaluation."
        )

    pipeline.load_vectorstore(vectorstore_path)
    evaluator = Evaluator(pipeline)

    # ── 3. Run proxy-metric evaluation ────────────────────────────────────────
    evaluator.run_evaluation(verbose=True)

    # ── 4. Run RAGAS evaluation ───────────────────────────────────────────────
    evaluator.run_ragas_evaluation()

    # ── 5. Export combined results ────────────────────────────────────────────
    evaluator.export_results("evaluation_results.json")
