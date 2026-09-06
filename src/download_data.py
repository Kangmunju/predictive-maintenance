# python src/download_data.py

"""
공개 데이터셋 다운로드
======================
    python src/download_data.py

data/raw/ 는 .gitignore에 들어 있습니다(용량·저작권).
이 스크립트가 있으면 누구나 같은 상태를 재현할 수 있습니다.
★ 포트폴리오에서 "재현 가능한가"는 생각보다 크게 봅니다.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    "ai4i": {
        "url": "https://archive.ics.uci.edu/static/public/601/"
        "ai4i+2020+predictive+maintenance+dataset.zip",
        "expect": ["ai4i2020.csv"],
    },
    "secom": {
        "url": "https://archive.ics.uci.edu/static/public/179/secom.zip",
        "expect": ["secom.data", "secom_labels.data", "secom.names"],
    },
}


def fetch(name: str, spec: dict) -> bool:
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f).exists() for f in spec["expect"]):
        print(f"[skip] {name} — 이미 있습니다")
        return True

    zpath = RAW / f"{name}.zip"
    print(f"[get ] {name} ← {spec['url']}")
    try:
        r = requests.get(
            spec["url"], timeout=120, headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        print("       UCI 서버가 가끔 느립니다. 잠시 후 다시 시도하거나")
        print("       브라우저로 직접 받아 data/raw/ 아래에 푸세요.")
        return False

    zpath.write_bytes(r.content)
    md5 = hashlib.md5(r.content).hexdigest()
    print(f"       {len(r.content):,} bytes  md5={md5}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    missing = [f for f in spec["expect"] if not (out / f).exists()]
    if missing:
        print(f"[WARN] 기대한 파일이 없습니다: {missing}")
        print(f"       실제 내용: {[p.name for p in out.iterdir()]}")
        return False
    print(f"[ok  ] {name} → {out}")
    return True


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = all(fetch(n, s) for n, s in SOURCES.items())
    if ok:
        print("\n전부 준비됐습니다. 이제 analysis/ 의 스크립트를 돌리면 됩니다.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
