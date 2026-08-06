"""Interactive command-line demo.

    python -m scripts.demo

Type a question, get a grounded answer plus its sources. Ctrl-C to quit.
"""
from __future__ import annotations

from src.pipeline import RAGPipeline


def main() -> None:
    print("Loading pipeline (first run downloads models)...")
    pipeline = RAGPipeline()
    print("Ready. Ask a question (Ctrl-C to quit).\n")
    try:
        while True:
            q = input("you > ").strip()
            if not q:
                continue
            result = pipeline.answer(q)
            print(f"\nassistant > {result.answer}\n")
            print("sources:")
            for i, s in enumerate(result.sources, 1):
                preview = s.text[:120].replace("\n", " ")
                print(f"  [{i}] {s.source} (score={s.score:.2f}) {preview}...")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
