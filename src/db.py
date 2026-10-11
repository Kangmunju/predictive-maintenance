"""
SQLite 적재 계층
================
설비 데이터는 "같은 설비 + 같은 관측 시각"이 유일해야 합니다.

기존 데이터는 (machine_id, ts)를 기준으로 저장되었습니다.
그러나 ts에는 센서 타임스탬프 흔들림이 적용될 수 있으므로,
신규 데이터는 (machine_id, observed_at)을 기준으로 중복을 방지합니다.

기존 데이터와 수집 로그를 유지하면서 스키마를 이전합니다.
"""
# 같은 설비 + 같은 관측 기준 시각 -> DB에 한 번만 존재해야 함
# ts는 센서가 전송한 시각, observed_at은 오염 전 관측 기준 시각임

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sensors.db"


# ----------------------------------------------------------------------
# DB 스키마
# ----------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id      TEXT    NOT NULL,
    ts              TEXT    NOT NULL,
    type            TEXT,
    air_temp_k      REAL,
    process_temp_k  REAL,
    rot_speed_rpm   REAL,
    torque_nm       REAL,
    tool_wear_min   REAL,
    vibration_mms   REAL,
    current_a       REAL,
    humidity_pct    REAL,
    machine_failure INTEGER,
    observed_at     TEXT,
    collected_at    TEXT
);

CREATE INDEX IF NOT EXISTS ix_sensor_ts
ON sensor_raw (ts);

CREATE INDEX IF NOT EXISTS ix_sensor_machine
ON sensor_raw (machine_id, ts);

CREATE UNIQUE INDEX IF NOT EXISTS ux_sensor_observed
ON sensor_raw (machine_id, observed_at)
WHERE observed_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS collect_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT,
    window_start  TEXT,
    window_end    TEXT,
    rows_received INTEGER,
    rows_inserted INTEGER,
    rows_skipped  INTEGER,
    note          TEXT
);
"""


COLUMNS = [
    "machine_id",
    "ts",
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
    "machine_failure",
    "observed_at",
    "collected_at",
]


# ----------------------------------------------------------------------
# 기존 DB 구조 확인
# ----------------------------------------------------------------------
def _table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()

    return row is not None


def _column_names(con: sqlite3.Connection, table_name: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _has_legacy_unique(con: sqlite3.Connection) -> bool:
    """
    기존 sensor_raw 테이블의 (machine_id, ts) UNIQUE 제약 확인.

    기존 테이블에 UNIQUE 제약이 남아 있으면
    신규 관측값의 ts가 충돌할 때 정상 행도 누락될 수 있음.
    """
    indexes = con.execute("PRAGMA index_list(sensor_raw)").fetchall()

    for index in indexes:
        index_name = index[1]
        is_unique = bool(index[2])

        if not is_unique:
            continue

        columns = con.execute(f'PRAGMA index_info("{index_name}")').fetchall()

        names = [column[2] for column in columns]

        if names == ["machine_id", "ts"]:
            return True

    return False


# ----------------------------------------------------------------------
# 기존 테이블 마이그레이션
# ----------------------------------------------------------------------
def _migrate_sensor_raw(con: sqlite3.Connection) -> None:
    """
    기존 sensor_raw 데이터를 보존하면서 테이블 구조를 변경함.

    - 기존 id 유지
    - 기존 ts 유지
    - 기존 collected_at 유지
    - 기존 행의 observed_at은 NULL
    - collect_log는 변경하지 않음
    """
    if not _table_exists(con, "sensor_raw"):
        return

    existing_columns = _column_names(con, "sensor_raw")
    needs_rebuild = _has_legacy_unique(con)

    if not needs_rebuild:
        if "observed_at" not in existing_columns:
            con.execute("ALTER TABLE sensor_raw ADD COLUMN observed_at TEXT")
        return

    # 기존 UNIQUE 제약은 ALTER TABLE만으로 제거할 수 없어
    # 새 테이블을 만들고 기존 데이터를 복사하는 방식으로 이전함.
    con.execute(
        """
        CREATE TABLE sensor_raw_new (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id      TEXT NOT NULL,
            ts              TEXT NOT NULL,
            type            TEXT,
            air_temp_k      REAL,
            process_temp_k  REAL,
            rot_speed_rpm   REAL,
            torque_nm       REAL,
            tool_wear_min   REAL,
            vibration_mms   REAL,
            current_a       REAL,
            humidity_pct    REAL,
            machine_failure INTEGER,
            observed_at     TEXT,
            collected_at    TEXT
        )
        """
    )

    if "observed_at" in existing_columns:
        observed_expr = "observed_at"
    else:
        observed_expr = "NULL"

    con.execute(
        f"""
        INSERT INTO sensor_raw_new (
            id,
            machine_id,
            ts,
            type,
            air_temp_k,
            process_temp_k,
            rot_speed_rpm,
            torque_nm,
            tool_wear_min,
            vibration_mms,
            current_a,
            humidity_pct,
            machine_failure,
            observed_at,
            collected_at
        )
        SELECT
            id,
            machine_id,
            ts,
            type,
            air_temp_k,
            process_temp_k,
            rot_speed_rpm,
            torque_nm,
            tool_wear_min,
            vibration_mms,
            current_a,
            humidity_pct,
            machine_failure,
            {observed_expr},
            collected_at
        FROM sensor_raw
        """
    )

    old_count = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]

    new_count = con.execute("SELECT COUNT(*) FROM sensor_raw_new").fetchone()[0]

    if old_count != new_count:
        raise RuntimeError(
            f"DB 이전 중 행 수 불일치: 기존 {old_count}, 신규 {new_count}"
        )

    con.execute("DROP TABLE sensor_raw")
    con.execute("ALTER TABLE sensor_raw_new RENAME TO sensor_raw")


# ----------------------------------------------------------------------
# SQLite DB 연결
# ----------------------------------------------------------------------
def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(path)

    try:
        # 기존 데이터가 있는 경우 테이블 구조부터 안전하게 이전함
        # 마이그레이션과 인덱스 생성은 하나의 트랜잭션으로 처리함
        con.execute("BEGIN IMMEDIATE")

        _migrate_sensor_raw(con)

        # executescript()는 트랜잭션을 자동 커밋할 수 있으므로
        # 마이그레이션 이후에는 개별 SQL 문을 실행함
        for statement in SCHEMA.split(";"):
            statement = statement.strip()

            if statement:
                con.execute(statement)

        con.commit()

    except Exception:
        con.rollback()
        con.close()
        raise

    return con


# ----------------------------------------------------------------------
# DataFrame의 센서 데이터를 DB에 적재
# ----------------------------------------------------------------------
def upsert(con: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    """
    DataFrame의 센서 데이터를 DB에 저장함.

    신규 데이터:
        (machine_id, observed_at) 중복 방지

    기존 데이터:
        observed_at이 없으면 NULL로 저장

    반환값:
        (삽입된 행 수, 건너뛴 행 수)
    """
    df = df.reindex(columns=COLUMNS)

    sql = (
        f"INSERT OR IGNORE INTO sensor_raw ({','.join(COLUMNS)}) "
        f"VALUES ({','.join('?' * len(COLUMNS))})"
    )

    # 숫자형 컬럼의 NaN도 SQL NULL로 저장되도록
    # object 타입으로 변환한 뒤 None을 적용함
    clean_df = df.astype(object).where(pd.notna(df), None)

    records = list(clean_df.itertuples(index=False, name=None))

    before = con.total_changes

    con.executemany(sql, records)
    con.commit()

    inserted = con.total_changes - before
    skipped = len(records) - inserted

    return inserted, skipped


# ----------------------------------------------------------------------
# 수집 실행 이력 기록
# ----------------------------------------------------------------------
def log_run(
    con: sqlite3.Connection,
    window_start,
    window_end,
    received,
    inserted,
    skipped,
    note="",
):
    con.execute(
        "INSERT INTO collect_log (run_at, window_start, window_end,"
        " rows_received, rows_inserted, rows_skipped, note)"
        " VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (
            str(window_start),
            str(window_end),
            received,
            inserted,
            skipped,
            note,
        ),
    )
    con.commit()


# ----------------------------------------------------------------------
# 저장된 센서 데이터 조회
# ----------------------------------------------------------------------
def read_all(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM sensor_raw ORDER BY ts, machine_id",
        con,
    )


# =============================================================================
# [구현 핵심]
# =============================================================================

# SQLite를 사용해 설비별 센서 데이터를 저장하는 DB 적재 계층을 구현함
# sensor_raw에 센서 데이터, collect_log에 수집 실행 이력을 저장함
# 기존에는 (machine_id, ts)에 UNIQUE 제약을 적용해 중복 저장을 방지함
# INSERT OR IGNORE를 사용해 중복 데이터가 들어오더라도 기존 행을 유지함
# upsert()에서 삽입된 행 수와 건너뛴 행 수를 계산하도록 구현함

# 센서 타임스탬프 흔들림으로 서로 다른 관측값의 ts가 충돌할 수 있는 문제를 확인함
# 실제 관측 기준 시각인 observed_at을 추가하고 신규 데이터의 중복 기준으로 변경함
# 기존 데이터와 ID를 유지하면서 테이블 구조를 이전하는 마이그레이션을 추가함
# 기존 데이터의 observed_at은 복원할 수 없으므로 NULL로 유지함
# (machine_id, observed_at)에 UNIQUE 인덱스를 적용해 신규 데이터 중복을 방지함
# 숫자형 NaN을 SQL NULL로 저장하도록 결측값 변환 방식을 수정함
# collect_log와 기존 수집 이력을 유지하도록 구현함

# 추후 보완 사항
# 1. 기존 데이터와 신규 데이터 사이의 중복 관계를 판단하는 기준 보완
# 2. observed_at이 없는 과거 데이터의 관측 시각 복원 가능성 검토
# 3. 대용량 DB에서 테이블 재구성 시 소요 시간 및 저장 공간 검증
# 4. collector.py의 날짜별 CSV 저장 및 공백 복구 로직과 연동 검증
