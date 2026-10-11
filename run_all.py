"""
전체 재현 스크립트
==================
    python run_all.py
    python run_all.py --offline

기본 실행:
데이터 다운로드 → 수집 → DB → 학습 → 분석 a1~a7 을 순서대로 실행합니다.

오프라인 실행:
기존 CSV 데이터 → DB → 학습 → 분석 a1~a7 을 순서대로 실행합니다.
새로운 데이터 다운로드와 센서 수집은 수행하지 않습니다.

각 단계의 출력은 outputs/ 아래에 저장됩니다.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# 기존 전체 실행 단계
# 기본 실행에서는 데이터 다운로드와 24시간 센서 수집을 포함합니다.
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


def parse_args():
    """
    명령줄 옵션을 읽습니다.

    --offline:
        새로운 데이터 다운로드와 센서 수집을 생략합니다.
        이미 저장된 CSV를 이용해 DB·학습·분석을 다시 실행합니다.
    """

    parser = argparse.ArgumentParser(
        description="설비 예지보전 전체 파이프라인 재현 스크립트"
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="다운로드와 센서 수집을 생략하고 기존 데이터로 재현 실행",
    )

    return parser.parse_args()


def get_steps(offline: bool):
    """
    실행 모드에 따라 단계 목록을 구성합니다.

    기본 모드:
        STEPS 전체 실행

    오프라인 모드:
        데이터 다운로드와 센서 수집 단계를 제외

    기존 STEPS를 직접 변경하지 않고 새로운 목록을 반환합니다.
    """

    if not offline:
        return STEPS

    return [
        step for step in STEPS if step[0] not in ("데이터 다운로드", "센서 수집(24h)")
    ]


def main() -> int:
    args = parse_args()

    steps = get_steps(args.offline)

    # 실행 모드를 먼저 출력해
    # 새로운 데이터가 생성되는 실행인지 구분합니다.
    print("\n[전체 재현 실행]")

    if args.offline:
        print("모드: OFFLINE")
        print("새로운 데이터 다운로드 및 센서 수집을 생략합니다.")
        print("기존 CSV를 이용해 DB·학습·분석을 실행합니다.")

        # 오프라인 모드에서는 기존 데이터가 있어야 합니다.
        # 데이터가 없다면 DB 재생성 전에 명확한 오류를 출력합니다.
        history_files = list((ROOT / "data" / "history").glob("*.csv"))

        if not history_files:
            print(
                "오류: data/history/에 CSV 파일이 없습니다. "
                "오프라인 재현을 실행할 수 없습니다."
            )
            return 1

        print(f"기존 센서 CSV 파일: {len(history_files)}개")

    else:
        print("모드: DEFAULT")
        print("데이터 다운로드 및 24시간 센서 수집을 포함합니다.")
        print("기존 data/history/ 데이터에 새 데이터가 추가될 수 있습니다.")

    fails = []

    for name, cmd, logfile in steps:
        cwd = (
            ROOT / "analysis"
            if cmd[1].startswith("a") and cmd[1].endswith(".py")
            else ROOT
        )

        t0 = time.time()

        print(
            f"\n{'=' * 60}\n▶ {name}\n{'=' * 60}",
            flush=True,
        )

        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        out = r.stdout + r.stderr

        if logfile:
            (OUT / logfile).write_text(
                out,
                encoding="utf-8",
            )

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


# ============================================================
# [구현 핵심]
# ============================================================

# 데이터 다운로드 → 센서 수집 → DB 재생성 → 모델 학습 → 분석 a1~a7을
# 순서대로 실행하는 전체 재현 스크립트를 구현함
# subprocess로 각 단계를 실행하고 실행 시간과 성공·실패 여부를 기록함
# 학습 및 분석 결과를 outputs/ 아래의 텍스트 파일로 저장함
# 개별 단계가 실패해도 다음 단계를 실행하고 마지막에 실패 목록을 출력함

# argparse를 이용해 --offline 실행 옵션을 추가함
# 오프라인 실행에서는 데이터 다운로드와 센서 수집 단계를 제외함
# 기존 CSV를 이용한 DB 재생성·모델 학습·분석 과정은 그대로 유지함
# 오프라인 실행 전에 센서 CSV 파일 존재 여부를 확인하도록 추가함
# 실행 모드를 출력해 신규 데이터 수집 여부를 명확히 구분함
# subprocess 출력의 UTF-8 디코딩 오류를 방지하도록 개선함

# 추후 보완 사항
# 1. 분석 단계별 필수 데이터 파일의 존재 여부 사전 검증
# 2. 실행 결과와 입력 데이터의 해시값을 함께 기록하는 재현성 보고서 추가
# 3. 일부 단계 실패 시 후속 단계 실행 여부를 선택하는 옵션 검토
