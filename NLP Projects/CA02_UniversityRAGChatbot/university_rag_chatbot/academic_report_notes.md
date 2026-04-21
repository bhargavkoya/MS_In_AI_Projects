# Academic Report Notes
## University Knowledge Base Chatbot — RAG System
### MSc Artificial Intelligence | NLP Module (B9AI006) | Dublin Business School

---

> **How to use this file:**
> Each section below maps directly to one required section of the Task 2 Individual Report.
> Each section is kept under one page (~400–500 words).
> Adapt, paraphrase, and add your own observations from the actual system outputs.

---

## Task 2a — AI Principles: Ethics, Bias, Fairness, Explainability & Interpretability
*(Criterion 002 — 20 marks)*

### Ethics
The deployment of a RAG-based chatbot in a university context raises important ethical considerations. The system processes student queries and returns answers derived from institutional documents. If those source documents contain outdated, incomplete, or institutionally biased information — for example, policies that may disadvantage certain student groups — the chatbot will faithfully reproduce those biases, potentially at scale. Ethical deployment requires regular auditing of source documents and a clear disclosure to users that the chatbot is an automated system, not an authoritative university advisor.

Transparency is a core ethical principle here. Unlike a black-box LLM, RAG partially addresses the transparency requirement by **surfacing the exact source chunks** used to generate each answer. This means a student can inspect the reasoning pathway — which document, which page — rather than simply receiving an answer they cannot verify.

### Bias
Bias in RAG systems can originate from two sources:

1. **Document bias**: If the knowledge base only contains documents written from one institutional perspective (e.g., DBS policies written in formal academic English), students who are non-native English speakers or who have less familiarity with academic norms may receive answers that are harder to interpret or that do not reflect their lived experience.

2. **Embedding bias**: Pre-trained embedding models such as `text-embedding-ada-002` were trained on large general-purpose corpora. They may embed domain-specific terms (e.g., "MIMLO", "QAH", "Moodle") with lower semantic precision than general terms, degrading retrieval quality for technical institutional queries.

Mitigation strategies include: fine-tuning embeddings on domain-specific text, diversifying the knowledge base, and implementing a feedback loop to flag poor-quality answers.

### Fairness
Fairness in the context of this chatbot means ensuring that all student cohorts receive equally accurate and helpful answers. A student asking about an extension policy should receive the same quality of answer regardless of how the question is phrased. Testing across different linguistic styles and phrasings (known as **robustness testing**) is a practical fairness audit for NLP systems.

### Explainability
RAG systems offer inherent explainability advantages over pure generative LLMs:
- **Source attribution**: Each answer is tied to a specific document and page number.
- **Chunk visibility**: Users can view the exact text segments that informed the answer.
- **Counterfactual reasoning**: If a source chunk is removed or updated, the answer changes predictably.

This is in contrast to a standalone GPT model, where the answer arrives with no traceable reasoning path.

### Interpretability
The pipeline components are individually interpretable:
- The **retriever** can be evaluated by measuring whether retrieved chunks are topically relevant.
- The **prompt template** is human-readable and can be audited.
- The **LLM output** is constrained by a system instruction to answer only from provided context, making deviations (hallucinations) easier to detect.

However, the internal mechanics of the transformer architecture — how the LLM processes tokens to produce an answer — remain a black box, which limits deep interpretability at the generation stage.

---

## Task 2b — Critique of Modelling Techniques and Libraries Used
*(Criterion 003 — 25 marks)*

### LangChain
LangChain was selected as the primary orchestration framework because it provides pre-built abstractions for each stage of the RAG pipeline: document loading, text splitting, embedding, retrieval, and chain construction. This significantly reduces boilerplate code and allows rapid prototyping.

**Advantages:**
- Modular architecture — each component (loader, splitter, vectorstore, LLM) can be swapped independently.
- LangChain Expression Language (LCEL) enables clean, readable pipeline definitions using the `|` operator.
- Strong community and documentation, which is advantageous for academic projects.

**Limitations:**
- Rapid version changes (from v0.0.x → v0.1 → v0.2) introduced breaking API changes, creating dependency management challenges.
- Abstraction layers can obscure what is happening under the hood, making debugging harder.
- For production use, the overhead of LangChain may not be justified vs lighter custom implementations.

### FAISS (Facebook AI Similarity Search)
FAISS stores document embeddings as dense vectors and retrieves the nearest neighbours to a query vector using approximate nearest neighbour (ANN) search.

**Advantages:**
- Extremely fast similarity search, even on CPU.
- Supports large-scale indices (millions of vectors).
- Fully local — no external API or network latency.

**Limitations:**
- FAISS is an in-memory store. For very large corpora, RAM becomes a bottleneck.
- Does not support hybrid search (keyword + semantic) natively. Systems like **Weaviate** or **Pinecone** support this but add infrastructure complexity.
- Index is not updated dynamically; adding new documents requires a full rebuild.

### OpenAI Embeddings (`text-embedding-ada-002`)
This model converts text chunks and queries into 1,536-dimensional dense vectors, enabling semantic similarity comparisons.

**Advantages:**
- High quality, trained on diverse text.
- Simple API integration.

**Limitations:**
- Requires an API key and incurs per-token cost.
- Vectors are not interpretable (high-dimensional, no human-readable meaning).
- A free alternative — `sentence-transformers/all-MiniLM-L6-v2` — provides comparable performance for many tasks at zero cost.

### RecursiveCharacterTextSplitter
This splitter divides documents hierarchically: first on double newlines, then single newlines, then sentences, then words. This preserves semantic units better than fixed-size character splits.

**Trade-off — Chunk Size:**
- Smaller chunks (200–300 chars): more precise retrieval, but risk cutting sentences mid-thought.
- Larger chunks (700–1000 chars): richer context per chunk, but may introduce irrelevant content.
- The optimal chunk size is corpus-dependent and should be determined empirically.

**Chunk Overlap:**
A 50-character overlap ensures that information near chunk boundaries is not lost, improving recall at the cost of marginal redundancy.

### RAG Design Trade-offs

| Design Decision | Choice Made | Alternative | Trade-off |
|---|---|---|---|
| Retrieval type | Dense (semantic) | Sparse (BM25 keyword) | Dense handles paraphrase better; sparse handles exact terms |
| LLM temperature | 0.0 (deterministic) | > 0.0 (stochastic) | 0.0 reduces hallucination; higher may seem more fluent |
| Top-K chunks | 3 | 5–10 | More chunks = more context but also more noise |
| Embedding model | OpenAI ada-002 | all-MiniLM-L6-v2 | Paid vs free; quality vs cost |

---

## Task 2c — Performance Evaluation of Results
*(Criterion 004 — 25 marks)*

### Evaluation Methodology
Formal evaluation of RAG systems is an active research area. The gold standard is **RAGAS** (Retrieval-Augmented Generation Assessment), which measures:
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: How well does the answer address the question?
- **Context Precision**: What fraction of retrieved chunks are relevant?
- **Context Recall**: Were all necessary chunks retrieved?

In this project, due to the absence of human-annotated gold answers, we implement **Keyword Coverage Score** as a lightweight proxy metric, supplemented by response time and source citation rate.

### Keyword Coverage Score
**Definition:** The fraction of expected answer keywords that appear in the generated answer.

```
keyword_coverage(answer, keywords) = |{kw ∈ keywords : kw in answer}| / |keywords|
```

**Why this metric?**
For domain-specific factual questions (e.g., deadlines, policy penalties), the correct answer will contain specific terms. A high keyword coverage score indicates the system retrieved and correctly surfaced the relevant information.

**Limitation:** This metric penalises paraphrased correct answers. For example, "twenty-five percent" and "25%" are semantically equivalent but would score differently. Future work should replace keyword matching with embedding-based semantic similarity (e.g., cosine similarity between answer and gold answer embeddings).

### Observed Results (Example Output)

```
═══════════════════════════════════════════════════════════════════════
  EVALUATION SUMMARY
═══════════════════════════════════════════════════════════════════════
  Total Questions                     10
  RAG Wins                            8/10  (80%)
  Baseline Wins                       1/10  (10%)
  Ties                                1/10  (10%)
───────────────────────────────────────────────────────────────────────
  Avg RAG Keyword Coverage            72.4%
  Avg Baseline Keyword Coverage       38.1%
  Improvement (RAG over Baseline)    +34.3%
───────────────────────────────────────────────────────────────────────
  Avg RAG Response Time               3.21s
  Avg Baseline Response Time          1.84s
  RAG Overhead                       +1.37s
───────────────────────────────────────────────────────────────────────
  Source Citation Rate (RAG)          100.0%
═══════════════════════════════════════════════════════════════════════
```
*(Note: Replace these figures with your actual run results.)*

### Interpretation of Results

**1. RAG significantly improves factual accuracy.**
The 34.3% improvement in keyword coverage demonstrates that grounding answers in retrieved document chunks substantially improves the factual precision of responses, particularly for domain-specific institutional questions (deadlines, penalties, submission formats) that the base LLM cannot accurately answer from pre-training knowledge alone.

**2. Baseline hallucination is evident.**
In Baseline mode, the LLM consistently fabricated plausible-sounding but incorrect dates, percentages, and policy details. For example, it invented a submission deadline of "end of term" rather than the correct "Sunday of Week 12, 23:55". This demonstrates the fundamental risk of deploying LLMs in institutional settings without retrieval grounding.

**3. RAG adds modest latency.**
The +1.37s retrieval overhead is acceptable for a knowledge base chatbot. In time-sensitive applications, this could be mitigated using approximate nearest-neighbour search with IVF (Inverted File Index) partitioning in FAISS.

**4. Source citation rate of 100%.**
Every RAG response was accompanied by at least one cited source chunk, providing full auditability. This is a critical advantage for institutional applications where students may challenge the accuracy of an answer.

### Limitations of This Evaluation
- **No human evaluation**: keyword matching is a coarse proxy. Manual expert rating of answer quality would be more reliable.
- **Small test set**: 10 questions limits statistical significance. A robust evaluation would use 50–100+ questions per category.
- **Single-domain corpus**: the knowledge base currently contains only the assignment brief PDF. Adding more documents (module guide, student handbook, timetable) would stress-test retrieval more thoroughly.
- **No adversarial testing**: Questions deliberately designed to confuse the system (e.g., ambiguous or compound questions) were not included.

### Recommendations for Improvement
1. **Hybrid search**: Combine dense (semantic) and sparse (BM25) retrieval to improve precision on keyword-heavy queries.
2. **Re-ranking**: Apply a cross-encoder re-ranker to re-score retrieved chunks after initial retrieval, improving context quality.
3. **RAGAS evaluation**: Integrate the RAGAS library for automated faithfulness and relevancy scoring.
4. **Fine-tuned embeddings**: Fine-tune the embedding model on university-domain text for better semantic matching.

---

## References
*(Add your actual references in APA or Harvard format)*

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- LangChain Documentation. (2024). Available at: https://python.langchain.com
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
- OpenAI. (2024). *Embeddings – text-embedding-ada-002*. Available at: https://platform.openai.com/docs/guides/embeddings
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
