
import os, argparse, heapq, re
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEXT_DIR = os.path.join(DATA_DIR, "texts")

def score_text(text, query_terms):
    s = 0
    for term in query_terms:
        s += len(re.findall(re.escape(term), text, flags=re.IGNORECASE))
    return s

def search(query, top_k=5):
    if not os.path.exists(TEXT_DIR):
        print("No extracted texts found. Run scripts/ingest.py first to extract text from PDFs.")
        return
    query_terms = [t.strip() for t in query.split() if t.strip()]
    scores = []
    for fname in os.listdir(TEXT_DIR):
        path = os.path.join(TEXT_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        sc = score_text(txt, query_terms)
        if sc > 0:
            scores.append((sc, fname, txt))
    if not scores:
        print("No matches found.")
        return
    best = heapq.nlargest(top_k, scores, key=lambda x: x[0])
    for sc, fname, txt in best:
        print("="*80)
        print(f"File: {fname} | Score: {sc}")
        first_match = None
        for term in query_terms:
            m = re.search(re.escape(term), txt, flags=re.IGNORECASE)
            if m:
                first_match = m.start()
                break
        snippet = txt[first_match:first_match+800] if first_match is not None else txt[:800]
        print(snippet.replace("\\n", " ")[:2000])
        print("\\n")

def main():
    parser = argparse.ArgumentParser(description="Keyword search over extracted PDF texts")
    parser.add_argument("query", nargs="+", help="Search terms (quoted)")
    parser.add_argument("--topk", type=int, default=5, help="Top K results to show")
    args = parser.parse_args()
    search(" ".join(args.query), top_k=args.topk)

if __name__ == '__main__':
    main()
