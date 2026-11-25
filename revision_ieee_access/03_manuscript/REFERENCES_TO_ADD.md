# References to add or fix (R1-m5, R1-m6, R2-18)

Working list for the revision. Venues and identifiers below were checked against
publisher/DBLP/proceedings pages on 2026-11-25. Page ranges are included only
where the source displayed them; do not invent a missing range. Insert as
`b33` onward and renumber the bibliography after the final pass.

## Source and venue fixes (R1-m5)

| Current issue | Action |
|---|---|
| MiniLM used throughout but no source paper | Add Reimers & Gurevych, Sentence-BERT, EMNLP-IJCNLP 2019. |
| [28] (BERT) cited as the source for dense bi-encoders | Keep BERT for the encoder; cite Sentence-BERT for the bi-encoder recipe. |
| [30] (GPT-4 report) cited for RAG applied to ticket reuse | Replace with Lewis et al., RAG, NeurIPS 2020; cite the ticket-reuse paper actually intended separately. |
| [29] (Transformers library) cited for passing five candidates | Cite the library only for implementation; the top-5 context choice is a design decision. |
| arXiv-only entries [6], [30], [34] | Use venue versions where they exist; keep arXiv identifiers only for the rest. |
| Laplace lift called “standard” with no citation | Delete “standard”; call it the inherited configuration of the submitted study. |

## Verified entries

### LLM relevance judgments and judge bias

1. G. Faggioli, L. Dietz, C. L. A. Clarke, G. Demartini, M. Hagen, C. Hauff,
   N. Kando, E. Kanoulas, M. Potthast, I. Soboroff, B. Stein, and H. Wachsmuth,
   “Perspectives on Large Language Models for Relevance Judgment,” in *Proc. ACM
   SIGIR Int. Conf. Theory of Information Retrieval (ICTIR)*, 2023.
   *Checked: ICTIR 2023, Taipei; best-paper award.*
2. P. Thomas, S. Spielman, N. Craswell, and B. Mitra, “Large Language Models Can
   Accurately Predict Searcher Preferences,” in *Proc. 47th Int. ACM SIGIR Conf.
   Res. Dev. Inf. Retrieval (SIGIR)*, 2024. DOI: 10.1145/3626772.3657707.
3. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin,
   Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging
   LLM-as-a-Judge with MT-Bench and Chatbot Arena,” in *Proc. NeurIPS Datasets
   and Benchmarks Track*, 2023.
4. P. Wang, L. Li, L. Chen, D. Zhu, B. Lin, Y. Cao, Q. Liu, T. Liu, and Z. Sui,
   “Large Language Models are not Fair Evaluators,” in *Proc. ACL*, 2024.
   DOI: 10.18653/v1/2024.acl-long.511.
5. A. Panickssery, S. R. Bowman, and S. Feng, “LLM Evaluators Recognize and Favor
   Their Own Generations,” in *Proc. NeurIPS*, 2024.

### Counterfactual and unbiased learning to rank

6. T. Joachims, A. Swaminathan, and T. Schnabel, “Unbiased Learning-to-Rank with
   Biased Feedback,” in *Proc. 10th ACM Int. Conf. Web Search Data Mining
   (WSDM)*, 2017, pp. 781–789. DOI: 10.1145/3018661.3018699.

### Bayesian smoothing, intervals and few ratings

7. E. B. Wilson, “Probable Inference, the Law of Succession, and Statistical
   Inference,” *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927.
   DOI: 10.1080/01621459.1927.10502953.
8. B. Efron and C. Morris, “Stein's Estimation Rule and Its Competitors—An
   Empirical Bayes Approach,” *J. Amer. Statist. Assoc.*, vol. 68, no. 341,
   pp. 117–130, 1973. DOI: 10.1080/01621459.1973.10481362.
   *Use for the empirical-Bayes prior; the paper does not claim a new smoother.*
9. A production-ranking “Bayesian average” is a practice, not one canonical paper.
   If a citation is needed, cite the Efron–Morris empirical-Bayes construction and
   the Wilson interval; do not invent a “Bayesian average” source.

### RAG evaluation

10. S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAS: Automated
    Evaluation of Retrieval Augmented Generation,” in *Proc. EACL System
    Demonstrations*, 2024, pp. 150–158. DOI: 10.18653/v1/2024.eacl-demo.16.
11. J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “ARES: An Automated
    Evaluation Framework for Retrieval-Augmented Generation Systems,” in *Proc.
    NAACL*, 2024. DOI: 10.18653/v1/2024.naacl-long.20.
12. J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking Large Language Models in
    Retrieval-Augmented Generation,” in *Proc. AAAI*, 2024, vol. 38, no. 16,
    pp. 17754–17762. DOI: 10.1609/aaai.v38i16.29728.
13. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, “BEIR: A
    Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
    Models,” in *Proc. NeurIPS Datasets and Benchmarks Track*, 2021.

### Embedding and retrieval background

14. N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese
    BERT-Networks,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 3980–3990.
    DOI: 10.18653/v1/D19-1410.
15. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
    H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela,
    “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” in
    *Proc. NeurIPS*, 2020.
16. N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “MTEB: Massive Text
    Embedding Benchmark,” in *Proc. EACL*, 2023, pp. 2014–2037.
    DOI: 10.18653/v1/2023.eacl-main.148. *Optional; use only if the embedding
    choice is discussed.*

## Where each is used

| Section | Entries |
|---|---|
| 2.2 Learning to rank | 6 |
| 2.3 LLM-as-judge validity | 1–5 |
| 2.4 Calibration and few ratings | 7–9 |
| 2.7 RAG evaluation | 10–13 |
| 3/4 Models and retrieval | 14–16 |
| 1 Introduction | 14, 15 (source and pipeline) |

## What not to do

- Do not cite a reference the sentence does not support, even if a reviewer
  suggested the topic.
- Do not add a citation for the “standard Laplace lift”; no source was found.
- Do not list all sixteen if only a few are used. The final list should be
  proportionate and every entry should appear in the text.
- Re-check the final page ranges and DOIs against the publisher page after the
  bibliography is assembled; proceedings pagination can differ from the preprint.
