
import os, argparse, json, re, heapq
try:
    import openai
except Exception:
    openai = None
try:
    import faiss
except Exception:
    faiss = None
try:
    import numpy as np
except Exception:
    np = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
INDEX_DIR = os.path.join(DATA_DIR, "index_store")

def load_texts():
    texts = []
    names = []
    if not os.path.exists(TEXT_DIR):
        raise RuntimeError("No extracted texts found. Run scripts/ingest.py first to extract text from PDFs.")
    for fname in os.listdir(TEXT_DIR):
        path = os.path.join(TEXT_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        texts.append(txt)
        names.append(fname)
    return names, texts

def make_embeddings(texts, api_key_env="OPENAI_API_KEY", model="text-embedding-3-small"):
    if openai is None:
        raise RuntimeError("openai package not installed.")
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"OpenAI API key not set in environment variable {api_key_env}.")
    openai.api_key = key
    embs = []
    BATCH = 8
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = openai.Embedding.create(model=model, input=batch)
        for d in resp.data:
            embs.append(d.embedding)
    return embs

def build_faiss_index(names, texts, index_path=INDEX_DIR):
    if faiss is None or np is None:
        raise RuntimeError("faiss or numpy not installed. Install faiss-cpu and numpy.")
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    print("Creating embeddings (this calls OpenAI Embeddings API)...")
    embs = make_embeddings(texts)
    vecs = np.array(embs).astype('float32')
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, os.path.join(index_path, "faiss.index"))
    # save metadata
    meta = {"names": names}
    with open(os.path.join(index_path, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(f"Saved index to {index_path} (n={len(names)})")

def load_faiss_index(index_path=INDEX_DIR):
    if faiss is None or np is None:
        raise RuntimeError("faiss or numpy not installed.")
    if not os.path.exists(os.path.join(index_path, "faiss.index")):
        return None
    index = faiss.read_index(os.path.join(index_path, "faiss.index"))
    with open(os.path.join(index_path, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return index, meta

def retrieve_semantic(query, top_k=3, model="text-embedding-3-small", api_key_env="OPENAI_API_KEY"):
    idx = load_faiss_index()
    if idx is None:
        raise RuntimeError("Index not found. Run build-index first.")
    index, meta = idx
    if openai is None:
        raise RuntimeError("openai package not installed.")
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"OpenAI API key not set in environment variable {api_key_env}.")
    openai.api_key = key
    resp = openai.Embedding.create(model=model, input=[query])
    qvec = np.array(resp.data[0].embedding).astype('float32').reshape(1, -1)
    D, I = index.search(qvec, top_k)
    results = []
    for i in I[0]:
        name = meta["names"][i]
        path = os.path.join(TEXT_DIR, name)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        results.append((name, txt[:1500]))
    return results

def ask_openai_with_context(question, contexts, api_key_env="OPENAI_API_KEY", model="gpt-4o-mini"):
    if openai is None:
        raise RuntimeError("openai package not installed.")
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"OpenAI API key not set in environment variable {api_key_env}.")
    openai.api_key = key
    joined = "\\n---\\n".join([f"File: {fn}\\n{sn}" for fn, sn in contexts])
    prompt = f\"\"\"You are an assistant answering questions about industrial safety documents. Use the context below when relevant.

Context:
{joined}

Question: {question}
\"\"\"
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a helpful assistant specialized in industrial safety."},
                      {"role":"user","content":prompt}],
            max_tokens=800,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Build/load FAISS index and ask questions (uses OpenAI embeddings + chat)")
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('build-index', help='Build FAISS index from extracted texts (calls OpenAI Embeddings API)')
    q = sub.add_parser('ask', help='Ask a question (uses semantic retrieval + chat)')
    q.add_argument('question', nargs='+', help='Question text')
    q.add_argument('--model', default='gpt-4o-mini', help='OpenAI chat model to use')
    parser.add_argument('--topk', type=int, default=3, help='Number of contexts to retrieve')
    args = parser.parse_args()
    if args.cmd == 'build-index':
        names, texts = load_texts()
        build_faiss_index(names, texts)
    elif args.cmd == 'ask':
        question = ' '.join(args.question)
        contexts = retrieve_semantic(question, top_k=args.topk)
        print('Retrieved contexts:')
        for fn, sn in contexts:
            print(f'-- {fn}: {sn[:200].replace(\"\\n\",\" \")}')
        print('\\nAsking LLM...')
        ans = ask_openai_with_context(question, contexts, model=args.model)
        print('\\n=== Answer ===\\n')
        print(ans)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
