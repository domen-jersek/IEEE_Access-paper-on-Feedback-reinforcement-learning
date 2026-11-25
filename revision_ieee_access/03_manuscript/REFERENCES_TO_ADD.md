# References to add or fix (R1-m5, R1-m6, R2-18)

Tentative entries. **Verify every bibliographic detail before inserting**; do not
copy DOIs from memory. The goal is to place the work, not to pad the list.

## Source and venue fixes (R1-m5)

| Current issue | Action |
|---|---|
| MiniLM used throughout but no source paper | Add Reimers & Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks”, EMNLP-IJCNLP 2019. |
| [28] (BERT) cited as the source for dense bi-encoders | Keep BERT for the encoder, add Sentence-BERT for the bi-encoder recipe. |
| [30] (GPT-4 report) cited for RAG applied to ticket reuse | Replace with Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”, NeurIPS 2020, plus the ticket-reuse paper actually intended. |
| [29] (Transformers library) cited for passing five candidates | Cite the library only for implementation; the top-5 context choice is a design decision, not a Transformers feature. |
| arXiv-only entries [6], [30], [34] | Add venue versions where they exist; keep arXiv identifiers for the rest. |
| Laplace popularity lift called “standard” with no citation | Either cite a concrete source for the construction or delete “standard”. The revision treats it as the submitted legacy cell, not as common practice. |

## Missing literatures (R1-m6, R2-18)

### LLM relevance judgments and agreement with humans

- Faggioli et al., “Perspectives on Large Language Models for Relevance Judgment”,
  ICTIR 2023.
- Thomas et al., “Large Language Models Can Accurately Predict Searcher
  Preferences”, SIGIR 2024.
- Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”, NeurIPS
  2023.
- Wang et al., “Large Language Models are not Fair Evaluators”, ACL 2024.
- Panickssery et al., “LLM Evaluators Recognize and Favor Their Own Generations”,
  NeurIPS 2024.

### Counterfactual and unbiased learning to rank

- Joachims, Swaminathan, Schnabel, “Unbiased Learning-to-Rank with Biased Feedback”,
  WSDM 2017.
- A recent survey of counterfactual learning to rank (verify the current standard
  reference before citing).

### Bayesian average, Wilson intervals, few ratings

- Wilson, “Probable Inference, the Law of Succession, and Statistical Inference”,
  JASA 1927.
- A Bayesian-average or Wilson-interval ranking reference used in production ranking
  systems (verify; the original “Bayesian average” is a practice, not a single paper).

### RAG evaluation

- Es et al., “RAGAS: Automated Evaluation of Retrieval Augmented Generation”, EACL
  2024 demo.
- Saad-Falcon et al., “ARES: An Automated Evaluation Framework for
  Retrieval-Augmented Generation Systems”, NAACL 2024.
- Chen et al., “Benchmarking Large Language Models in Retrieval-Augmented
  Generation”, AAAI 2024.
- Thakur et al., “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models”, NeurIPS 2021 datasets track (for retrieval
  evaluation practice).

### Embedding and retrieval background (already partly present)

- Reimers & Gurevych 2019 (above).
- Muennighoff et al., “MTEB: Massive Text Embedding Benchmark”, EACL 2023 (optional,
  for the choice of embedding families).

## What not to do

- Do not cite a reference the sentence does not support, even if a reviewer suggested
  the topic.
- Do not add a citation for “standard Laplace lift” unless a source is found; the
  revision can say “the legacy configuration used in the submitted study”.
- Keep the total reference list proportionate; the added entries should each be used
  in the text.
