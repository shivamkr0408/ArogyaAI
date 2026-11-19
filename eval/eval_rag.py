"""
Retrieval evaluation for the RAG layer.

The explanation shown to the user is grounded in chunks pulled from
`rag/knowledge_base/`. If retrieval returns the wrong disease's document,
the explanation is fluent and wrong -- the worst failure mode a grounded
system has. So retrieval gets measured on its own, with the standard IR
metrics, rather than being assumed to work because the output reads well.

Metrics:

* **Hit@k**   -- was a chunk from the correct disease in the top k?
* **MRR**     -- 1/rank of the first correct chunk, averaged. Rewards
                 putting the right document first, not just somewhere.
* **nDCG@5**  -- position-discounted gain, so a correct chunk at rank 1
                 counts for more than one at rank 5.
* **Precision@5** -- what fraction of returned chunks were on-topic.

Two retrieval modes are compared, because the production code filters by
disease slug once the classifier has a prediction:

* **unfiltered** -- pure semantic search over the whole corpus. This is the
  honest measure of the embedding model on this corpus.
* **filtered**   -- the production path, where the predicted disease
  constrains the search. Near-perfect by construction, so it is reported
  separately and not presented as a retrieval result.
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

RESULTS_DIR = os.path.join(HERE, "results")
GOLD_PATH = os.path.join(HERE, "gold", "rag_gold.json")

K_VALUES = (1, 3, 5)
TOP_K = 5


def load_gold():
    with open(GOLD_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["queries"]


def dcg(relevances) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances, k) -> float:
    actual = dcg(relevances[:k])
    ideal = dcg(sorted(relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def evaluate(queries, retrieve_fn, label):
    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    ndcgs = []
    precisions = []
    failures = []

    for case in queries:
        expected = case["expected_slug"]
        results = retrieve_fn(case, TOP_K)
        slugs = [r.get("disease_slug", "") for r in results]
        relevances = [1 if s == expected else 0 for s in slugs]

        for k in K_VALUES:
            if any(relevances[:k]):
                hits[k] += 1

        rank = next((i + 1 for i, rel in enumerate(relevances) if rel), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        ndcgs.append(ndcg_at_k(relevances, 5))
        precisions.append(sum(relevances) / len(relevances) if relevances else 0.0)

        if not any(relevances):
            failures.append(
                {
                    "query": case["query"],
                    "expected": expected,
                    "retrieved": slugs,
                }
            )

    n = len(queries) or 1
    return {
        "mode": label,
        "n_queries": len(queries),
        **{f"hit_at_{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "ndcg_at_5": round(sum(ndcgs) / n, 4),
        "precision_at_5": round(sum(precisions) / n, 4),
        "failures": failures,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from app.services import retriever

    print("=" * 78)
    print("RAG retrieval evaluation")
    print("=" * 78)
    print("Loading embedding model and Chroma collection...")
    if not retriever.init_retriever() or not retriever.is_ready():
        print("\nRetriever not ready. Run `python rag/build_index.py` first.")
        return

    queries = load_gold()
    print(f"Gold queries: {len(queries)}")

    collection = retriever._collection
    model = retriever._model
    print(f"Indexed chunks: {collection.count()}")

    def retrieve_unfiltered(case, k):
        embedding = model.encode([case["query"]]).tolist()
        raw = collection.query(query_embeddings=embedding, n_results=k)
        metadatas = (raw.get("metadatas") or [[]])[0]
        return [{"disease_slug": m.get("disease_slug", "")} for m in metadatas]

    def retrieve_filtered(case, k):
        # The production path: the classifier's prediction narrows the search.
        chunks = retriever.retrieve(case["expected_slug"], case["query"], k=k)
        return [{"disease_slug": c.get("disease_slug", "")} for c in chunks]

    start = time.perf_counter()
    unfiltered = evaluate(queries, retrieve_unfiltered, "unfiltered semantic search")
    filtered = evaluate(queries, retrieve_filtered, "filtered by predicted disease")
    elapsed = round(time.perf_counter() - start, 2)

    # Precision@5 has a hard ceiling here: each disease document only yields
    # a couple of chunks, so at most that many of five results can ever be
    # relevant. Reporting the ceiling stops 0.33 being read as a failure.
    chunks_per_disease = {}
    for case in queries:
        slug = case["expected_slug"]
        if slug not in chunks_per_disease:
            got = collection.get(where={"disease_slug": slug})
            chunks_per_disease[slug] = len((got or {}).get("ids") or [])
    ceiling = sum(
        min(chunks_per_disease[c["expected_slug"]], TOP_K) / TOP_K for c in queries
    ) / (len(queries) or 1)
    unfiltered["precision_at_5_ceiling"] = round(ceiling, 4)
    unfiltered["precision_at_5_vs_ceiling"] = (
        round(unfiltered["precision_at_5"] / ceiling, 4) if ceiling else None
    )
    unfiltered["avg_chunks_per_disease"] = round(
        sum(chunks_per_disease.values()) / (len(chunks_per_disease) or 1), 2
    )

    for report in (unfiltered, filtered):
        print(f"\n{report['mode']}  (n={report['n_queries']})")
        print(f"   Hit@1         : {report['hit_at_1']:.3f}")
        print(f"   Hit@3         : {report['hit_at_3']:.3f}")
        print(f"   Hit@5         : {report['hit_at_5']:.3f}")
        print(f"   MRR           : {report['mrr']:.3f}")
        print(f"   nDCG@5        : {report['ndcg_at_5']:.3f}")
        print(f"   Precision@5   : {report['precision_at_5']:.3f}")
        if "precision_at_5_ceiling" in report:
            print(f"      ceiling    : {report['precision_at_5_ceiling']:.3f} "
                  f"({report['avg_chunks_per_disease']} chunks/disease, so at most "
                  f"that many of {TOP_K} can be relevant)")
            print(f"      vs ceiling : {report['precision_at_5_vs_ceiling']:.3f}")

    if unfiltered["failures"]:
        print("\n" + "-" * 78)
        print("Unfiltered retrieval misses (correct disease not in top 5)")
        print("-" * 78)
        for failure in unfiltered["failures"]:
            print(f"   {failure['query']}")
            print(f"      expected  : {failure['expected']}")
            print(f"      retrieved : {', '.join(failure['retrieved'])}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_model": "all-MiniLM-L6-v2",
        "indexed_chunks": collection.count(),
        "chunk_size_words": 250,
        "chunk_overlap_words": 40,
        "top_k": TOP_K,
        "unfiltered": unfiltered,
        "filtered": filtered,
        "elapsed_s": elapsed,
        "note": (
            "The filtered mode is the production path: once the classifier "
            "names a condition, retrieval is constrained to that disease's "
            "chunks, so it is near-perfect by construction and is not a "
            "measure of retrieval quality. The unfiltered row is the honest "
            "measure of the embedding model on this corpus."
        ),
    }
    out_path = os.path.join(RESULTS_DIR, "rag_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {out_path}  ({elapsed}s)")


if __name__ == "__main__":
    main()
