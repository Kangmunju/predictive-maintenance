"""
2024년 고정 기간 시뮬레이션 데이터 생성
======================================
    python scripts/make_sim2024.py

기본 기간:
    2024-01-01 ~ 2024-01-14 (14일)

기존 collector.py의 --minutes와 --end 옵션을 이용해
날짜별 센서 데이터를 생성합니다.

기존 data/history/와 data/sensors.db를 변경하지 않도록
별도의 CSV 폴더와 SQLite DB를 사용합니다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2024, 1, 14)

OUTPUT_DIR = ROOT / "data" / "sim2024"
HISTORY_DIR = OUTPUT_DIR / "history"
DB_PATH = OUTPUT_DIR / "sensors.db"


def parse_args():
    """생성할 날짜 범위와 실행 옵션을 읽습니다."""

    parser = argparse.ArgumentParser(
        description="2024년 고정 기간의 시뮬레이션 데이터 생성"
    )

    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help="시작 날짜 YYYY-MM-DD (기본: 2024-01-01)",
    )

    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=DEFAULT_END,
        help="마지막 날짜 YYYY-MM-DD (기본: 2024-01-14, 포함)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 데이터를 생성하지 않고 실행 명령만 확인",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.start > args.end:
        print("[ERROR] 시작 날짜는 종료 날짜보다 늦을 수 없습니다.")
        return 1

    print("[2024년 시뮬레이션 데이터 생성]")
    print(f"시작 날짜: {args.start}")
    print(f"종료 날짜: {args.end}")
    print(f"CSV 저장 위치: {HISTORY_DIR}")
    print(f"DB 저장 위치: {DB_PATH}")

    if args.dry_run:
        print("모드: DRY RUN (파일을 생성하지 않음)")
    else:
        print("모드: 실제 데이터 생성")
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    current = args.start
    completed = 0

    while current <= args.end:
        # sample_window()는 --end 시각을 포함하지 않으므로
        # 생성할 날짜의 다음 날 00:00을 종료 시각으로 지정합니다.
        next_day = current + timedelta(days=1)

        end_time = next_day.strftime("%Y-%m-%d %H:%M:%S")

        cmd = [
            sys.executable,
            "src/collector.py",
            "--minutes",
            "1440",
            "--end",
            end_time,
            "--hist",
            str(HISTORY_DIR),
            "--db",
            str(DB_PATH),
        ]

        print(f"\n▶ {current} 데이터 생성")
        print("실행:", subprocess.list2cmdline(cmd))

        if not args.dry_run:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                check=False,
            )

            if result.returncode != 0:
                print(f"[ERROR] {current} 데이터 생성 실패")
                return 1

        completed += 1
        current = next_day

    print(f"\n[완료] 처리 날짜: {completed}일")

    if args.dry_run:
        print("실제 CSV와 DB는 생성하지 않았습니다.")
    else:
        print(f"CSV: {HISTORY_DIR}")
        print(f"DB: {DB_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------------------
# [구현 핵심]
# ----------------------------------------------------------------------

# 기존 collector.py의 수집 기능을 재사용하는 날짜별 시뮬레이션 생성 스크립트 구현
# 2024-01-01부터 2024-01-14까지 고정된 14일간의 센서 데이터를 생성하도록 구성
# 날짜별 종료 시각을 다음 날 00:00으로 지정하고 1440분씩 수집하도록 구현
# --start와 --end 옵션으로 필요한 생성 기간을 변경할 수 있도록 구성

# 기존 data/history/와 data/sensors.db를 변경하지 않도록 별도 저장 경로 적용
# --dry-run 옵션으로 실제 파일 생성 없이 실행 명령을 검증하도록 추가
# 날짜별 수집 실패 시 오류를 출력하고 실행을 중단하도록 처리

# 실제 14일 데이터 생성 결과 CSV 14개, 총 59,400행 확인
# 동일 기간 재실행 후에도 CSV 14개, 총 59,400행 유지 확인
# 재실행 후 SQLite sensor_raw 테이블의 59,400행 확인

# 추후 보완 사항
# 1. collected_at 등 실행 시각에 따라 달라지는 컬럼을 구분한 해시 검증
# 2. 재실행 전후 CSV와 SQLite의 데이터 내용 일치 여부 자동 검증
# 3. 날짜별 생성 행 수와 검증 결과를 기록하는 보고서 추가
