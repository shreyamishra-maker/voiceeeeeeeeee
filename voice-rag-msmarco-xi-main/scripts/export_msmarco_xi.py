"""Stream a selected MSMARCO-XI subset into normalized JSONL in data/."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from voicerag.data_loader import stream_msmarco_xi


def main() -> None:
    language = os.environ.get("HF_DATASET_CONFIG", "hi")
    split = os.environ.get("HF_DATASET_SPLIT", "train")
    limit = int(os.environ.get("HF_DATASET_LIMIT", "1000")) or None
    output = Path(os.environ.get(
        "HF_DATASET_OUTPUT",
        str(_ROOT / "data" / f"msmarco_xi_{language}_{split}.jsonl"),
    ))
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming MSMARCO-XI/{language} ({split})")
    print(f"Writing normalized records to {output}")
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in stream_msmarco_xi(split=split, language=language, limit=limit):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"Exported {count} records...")

    print(f"[SUCCESS] Exported {count} records to {output}")


if __name__ == "__main__":
    main()
