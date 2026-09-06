"""분석 스크립트 공통 설정 (한글 폰트, 그림 저장 경로, 경로 등록)"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw"
IMG_DIR = Path(os.environ.get("PD_IMG_DIR", ROOT / "outputs" / "figures"))
IMG_DIR.mkdir(parents=True, exist_ok=True)


def set_font():
    """OS별 한글 폰트. 없으면 경고만 내고 계속 진행합니다."""
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    else:
        print("[WARN] 한글 폰트를 찾지 못했습니다. 그래프 라벨이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name: str):
    path = IMG_DIR / f"pd_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[fig] {path.name}")
    return path


def rule(title: str = ""):
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


set_font()
