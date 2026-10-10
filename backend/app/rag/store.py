"""Policy RAG for the MVP: BM25 over heading-delimited chunks of markdown policy documents.

No embedding model is loaded (3 GB RAM). The interface (`PolicyStore.search`) is what a Qdrant-backed
vector store would implement later.
"""
import math
import re
from collections import Counter
from pathlib import Path

from ..core.config import settings

STOP = set("the a an of to in on is are was were for and or what whats how who whom which with by at from as be been it its this that do does did when where why while then than there their they will shall must can may have has had made make happen happens if so not no".split())


def stem(w: str) -> str:
    """Very light suffix stripping so dismiss/dismissal/dismissed or vehicle/vehicles match."""
    for suf in ("ations", "ation", "ings", "ing", "als", "al", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class PolicyStore:
    def __init__(self, directory: Path | None = None, k1=1.5, b=0.75):
        self.dir = Path(directory or settings.policies_dir)
        self.k1, self.b = k1, b
        self.chunks: list[dict] = []
        self.reload()

    def reload(self):
        self.chunks = []
        for f in sorted(self.dir.glob("*.md")):
            doc_title = ""
            sections = re.split(r"(?m)^#{1,3}\s+", f.read_text(encoding="utf8"))
            for sec in sections:
                sec = sec.strip()
                if not sec:
                    continue
                head, _, body = sec.partition("\n")
                text = (head + "\n" + body).strip()
                if len(body.strip()) < 20:  # bare title chunk: remember it as context for the doc's sections
                    doc_title = head.strip()
                    continue
                self.chunks.append({"source": f.name, "heading": head.strip(), "text": text,
                                    "tokens": Counter(tokenize(head * 2 + " " + doc_title * 2 + " " + body)),
                                    "doc": doc_title})
        self.N = len(self.chunks)
        self.avgdl = (sum(sum(c["tokens"].values()) for c in self.chunks) / self.N) if self.N else 0
        self.df = Counter(t for c in self.chunks for t in c["tokens"])

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokenize(query)
        scored = []
        for c in self.chunks:
            dl, score = sum(c["tokens"].values()), 0.0
            for t in q:
                f = c["tokens"].get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [{"source": c["source"], "heading": c["heading"], "text": c["text"], "score": round(s, 3)}
                for s, c in scored[:k]]
