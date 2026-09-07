"""
전체 재현 스크립트
==================
    python run_all.py

데이터 다운로드 → 수집 → DB → 학습 → 분석 a1~a7 을 순서대로 실행합니다.
각 단계의 출력은 outputs/ 아래에 저장됩니다.
"""

import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

STEPS = [
    ("데이터 다운로드", [sys.executable, "src/download_data.py"], None),
    ("센서 수집(24h)", [sys.executable, "src/collector.py", "--minutes", "1440"], None),
    ("DB 재생성", [sys.executable, "src/build_db.py"], None),
    ("모델 학습", [sys.executable, "src/train.py"], "train_output.txt"),
]
for name in [
    "a1_pipeline",
    "a2_ai4i",
    "a3_leakage",
    "a4_outlier_trap",
    "a5_drift",
    "a6_cost_threshold",
    "a7_secom",
]:
    STEPS.append(
        (f"분석 {name}", [sys.executable, f"{name}.py"], f"{name[:2]}_output.txt")
    )


def main() -> int:
    fails = []
    for name, cmd, logfile in STEPS:
        cwd = (
            ROOT / "analysis"
            if cmd[1].startswith("a") and cmd[1].endswith(".py")
            else ROOT
        )
        t0 = time.time()
        print(f"\n{'=' * 60}\n▶ {name}\n{'=' * 60}", flush=True)
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        if logfile:
            (OUT / logfile).write_text(out, encoding="utf-8")
        print(out[-1500:] if len(out) > 1500 else out)
        status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
        print(f"  → {status}  {time.time() - t0:.1f}s")
        if r.returncode != 0:
            fails.append(name)
    print(f"\n{'=' * 60}")
    print(f"완료. 실패 {len(fails)}건" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
