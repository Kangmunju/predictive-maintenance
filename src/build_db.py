"""
CSV 이력 → SQLite 재생성
========================
data/history/*.csv 만 있으면 센서 데이터 DB를 다시 만들 수 있습니다.
"DB는 산출물이고 CSV가 원본"이라는 구조입니다.

    python src/build_db.py

기본 실행은 임시 DB를 생성하고 검증한 뒤 결과를 출력합니다.
실제 sensors.db를 교체하려면 --replace 옵션을 사용합니다.

    python src/build_db.py --replace
"""

# collector가 만들어둔 일별 CSV들을 읽고 SQLite DB를 처음부터 다시 만들기 위해 작성한 파일
# 앞서 정한 규칙 그대로 DB 자체를 원본으로 보지 않고 CSV를 원본으로 볼 것
# 단, collect_log는 CSV에서 복원할 수 없으므로 기존 DB가 있다면 보존할 것

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db as dbmod  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"


# ----------------------------------------------------------------------
# 기존 DB의 수집 로그 보존
# ----------------------------------------------------------------------
def copy_collect_log(
    source_path: Path,
    target: sqlite3.Connection,
) -> int:
    """
    기존 DB의 collect_log를 새 DB에 복사함.

    CSV에는 센서 데이터만 저장되므로
    수집 실행 이력은 기존 DB에서 별도로 가져와야 함.
    """
    if not source_path.exists():
        return 0

    source = sqlite3.connect(source_path)

    try:
        exists = source.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'collect_log'
            """
        ).fetchone()

        if exists is None:
            return 0

        rows = source.execute(
            """
            SELECT
                id,
                run_at,
                window_start,
                window_end,
                rows_received,
                rows_inserted,
                rows_skipped,
                note
            FROM collect_log
            ORDER BY id
            """
        ).fetchall()

        target.executemany(
            """
            INSERT INTO collect_log (
                id,
                run_at,
                window_start,
                window_end,
                rows_received,
                rows_inserted,
                rows_skipped,
                note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        target.commit()
        return len(rows)

    finally:
        source.close()


# ----------------------------------------------------------------------
# CSV 이력으로 새 DB 생성
# ----------------------------------------------------------------------
def rebuild(
    files: list[Path],
    output_path: Path,
    existing_db: Path | None = None,
) -> tuple[int, int, int, int]:
    """
    CSV 전체를 읽어 새 SQLite DB를 생성함.

    반환값:
        (CSV 입력 행 수, DB 저장 행 수, 중복 행 수, 보존 로그 수)
    """
    con = dbmod.connect(output_path)

    total_in = 0
    total_ins = 0

    try:
        for f in files:
            df = pd.read_csv(f)

            # 기존 CSV에는 observed_at이 없을 수 있음
            # db.py의 upsert()가 없는 컬럼을 NULL로 처리함
            ins, skip = dbmod.upsert(con, df)

            total_in += len(df)
            total_ins += ins

            print(
                f"  {f.name:<20} 읽음 {len(df):>6,} / 신규 {ins:>6,} / 중복 {skip:>5,}"
            )

        # 기존 DB가 있다면 수집 실행 이력을 보존
        logs = 0

        if existing_db is not None:
            logs = copy_collect_log(existing_db, con)

        n = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]

        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(f"SQLite 무결성 검사 실패: {integrity}")

        if n != total_ins:
            raise RuntimeError(f"삽입 행 수 불일치: 누적 {total_ins}, DB {n}")

        return total_in, n, total_in - n, logs

    finally:
        con.close()


# ----------------------------------------------------------------------
# 실제 실행
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--hist",
        default=str(HIST),
        help="CSV 이력 폴더",
    )

    ap.add_argument(
        "--db",
        default=str(dbmod.DB_PATH),
        help="SQLite DB 경로",
    )

    ap.add_argument(
        "--replace",
        action="store_true",
        help="검증된 새 DB로 기존 DB 교체",
    )

    args = ap.parse_args()

    hist = Path(args.hist)
    dbp = Path(args.db)

    # data/history 폴더에서 확장자가 .csv인 파일을 모두 검색
    files = sorted(hist.glob("*.csv"))

    if not files:
        print(f"[WARN] {hist} 에 CSV가 없습니다. 먼저 collector.py를 돌리세요.")
        return 1

    dbp.parent.mkdir(parents=True, exist_ok=True)

    # 기존 DB를 삭제하지 않고 임시 DB에 먼저 재생성
    with tempfile.TemporaryDirectory(
        dir=dbp.parent,
        prefix="rebuild_",
    ) as tmp:
        temp_db = Path(tmp) / "sensors_new.db"

        print("[INFO] 임시 DB에 CSV 데이터 재생성 시작")

        total_in, n, duplicates, logs = rebuild(
            files,
            temp_db,
            existing_db=dbp if dbp.exists() else None,
        )

        print(f"\nCSV {len(files)}개 / 읽은 행 {total_in:,} / DB {n:,}행")

        print(f"중복으로 건너뛴 행 {duplicates:,}건")
        print(f"보존한 수집 로그 {logs:,}건")
        print("[OK] 임시 DB 재생성 및 무결성 검사 완료")

        if not args.replace:
            print("[INFO] 기존 DB는 변경하지 않았습니다.")
            print("[INFO] 실제 교체는 --replace 옵션으로 실행하세요.")
            return 0

        # 임시 DB의 검증이 완료된 경우에만 실제 DB 교체
        # 같은 디렉터리에서 os.replace()로 교체함
        os.replace(temp_db, dbp)

        print(f"[OK] DB 교체 완료: {dbp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------
# [가장 최근의 출력]

# total_ins = CSV를 넣으며 파이썬 내에서 직접 누적한 신규 저장 행 수
# n = 재생성 완료 후 SQLite DB를 조회한 실제 총 행 수
# 새 DB를 처음부터 만들었으므로 정상 처리되었다면 두 값이 일치해야 함

# [CSV -> DB 재생성 테스트 (기존 14일 검증)]
# collect.py 테스트 과정에서 생성된 2024-03-01.csv가 남아있어
# data/history 폴더에 총 15개의 CSV가 존재함을 확인
# 최초 계획한 14일(2주)치 데이터로 재생성을 검증하기 위해
# 2024-03-01.csv를 프로젝트 폴더 밖으로 임시 이동함

# PowerShell에서 Move-Item data\history\2024-03-01.csv ..\2024-03-01.csv 실행
# 이후 data/history에 14개의 CSV만 남은 것을 확인
# 당시 python src/build_db.py 실행 결과
# CSV 14개 / 읽은 행 55,591 / DB 55,591행
# 중복 0건
#
# 당시에는 기존 DB를 삭제한 후 CSV를 재적재하는 방식이었으며,
# 14일치 CSV만으로 동일한 센서 데이터를 재생성할 수 있음을 확인함

# 당시 build_db.py 전체 출력 결과
# 기존 DB 삭제 후 재생성합니다.
#   2024-01-01.csv       읽음  3,903 / 신규  3,903 / 중복     0
#   2024-01-02.csv       읽음  4,031 / 신규  4,031 / 중복     0
#   2024-01-03.csv       읽음  3,867 / 신규  3,867 / 중복     0
#   2024-01-04.csv       읽음  4,006 / 신규  4,006 / 중복     0
#   2024-01-05.csv       읽음  4,037 / 신규  4,037 / 중복     0
#   2024-01-06.csv       읽음  3,838 / 신규  3,838 / 중복     0
#   2024-01-07.csv       읽음  3,993 / 신규  3,993 / 중복     0
#   2024-01-08.csv       읽음  3,959 / 신규  3,959 / 중복     0
#   2024-01-09.csv       읽음  4,152 / 신규  4,152 / 중복     0
#   2024-01-10.csv       읽음  3,902 / 신규  3,902 / 중복     0
#   2024-01-11.csv       읽음  3,854 / 신규  3,854 / 중복     0
#   2024-01-12.csv       읽음  4,089 / 신규  4,089 / 중복     0
#   2024-01-13.csv       읽음  4,061 / 신규  4,061 / 중복     0
#   2024-01-14.csv       읽음  3,899 / 신규  3,899 / 중복     0
# CSV 14개 / 읽은 행 55,591 / DB 55,591행
# 차이 0행은 당시 (machine_id, ts) 중복 기준으로 계산됨

# =============================================================================
# [구현 핵심]
# =============================================================================

# CSV를 원본 데이터로 사용하고 SQLite DB를 재생성하는 파이프라인을 구현함
# 초기 구현에서는 기존 sensors.db를 삭제한 뒤 CSV를 순서대로 적재함
# 2024년 14일치 CSV 55,591행을 재생성하여 데이터 일치 여부를 검증함

# 기존 DB를 먼저 삭제하면 CSV 적재 실패 시 데이터와 수집 로그가 손실될 수 있는 문제를 확인함
# 임시 DB에 CSV를 먼저 적재하고 무결성을 검증한 뒤 교체하도록 변경함
# 기본 실행은 검증만 수행하고 기존 DB를 변경하지 않도록 구성함
# --replace 옵션을 사용할 때만 검증된 새 DB로 기존 DB를 교체함
# 기존 collect_log는 CSV로 복원할 수 없으므로 새 DB로 복사해 보존함

# 신규 데이터는 observed_at을 기준으로 중복을 방지하도록 DB 구조를 변경함
# 기존 CSV에는 observed_at이 없으므로 NULL로 적재하고 원본 값을 유지함

# [전체 CSV -> DB 재생성 및 교체 검증 (2026-10-11)]
# python3 src/build_db.py 실행
# CSV 44개 / 읽은 행 187,564 / 임시 DB 187,564행
# 중복 제외 0건 / 보존한 수집 로그 0건
# SQLite 무결성 검사: PASS
# 기존 DB를 변경하지 않고 임시 DB에서 재생성 가능함을 확인함

# 기존 DB의 55,591행이 CSV 원본에 모두 포함되어 있음을 별도로 검증함
# 기존 sensors.db를 sensors_before_rebuild.db로 백업한 후
# python3 src/build_db.py --replace 실행
# CSV 44개 / 읽은 행 187,564 / 실제 DB 187,564행
# 중복 제외 0건 / SQLite 무결성 검사: PASS
# 기존 DB를 전체 CSV 기준으로 교체하는 작업을 완료함

# 재생성 후 observed_at 컬럼을 확인한 결과
# 기존 CSV 전체가 과거 수집 형식이어서 observed_at은 187,564행 모두 NULL임
# 신규 simulator.py에서 observed_at이 정상 생성되는 것을 확인함
# collector.py 통합 테스트에서 최초 15행 저장 및 동일 구간 재수집 시 중복 15행 제외를 확인함
# 따라서 신규 수집 데이터부터 observed_at 기반 중복 방지 기능이 적용됨

# 추후 보완 사항
# 1. DB 교체 직전 자동 백업 및 장애 발생 시 복원 기능
# 2. 대용량 CSV를 청크 단위로 읽어 메모리 사용량을 줄이는 기능
# 3. CSV 파일 손상 및 필수 컬럼 누락 여부에 대한 사전 검증
# 4. 수집 로그를 별도 파일로 백업해 DB 없이도 재생성할 수 있는 구조
