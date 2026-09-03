"""
SQLite 적재 계층
================
설비 데이터는 "같은 시각 같은 설비"가 유일해야 합니다.
그래서 (machine_id, ts)에 UNIQUE 제약을 걸고 UPSERT로 넣습니다.
중복 전송이 와도 DB가 알아서 막아줍니다. 파이썬에서 막는 것보다 확실합니다.
"""
# 같은 설비 + 같은 시각 -> DB에 한 번만 존재해야 함(중복으로 보고 저장을 막음)

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sensors.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id      TEXT    NOT NULL,
    ts              TEXT    NOT NULL,          -- ISO8601 문자열 (UTC 기준)
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
    collected_at    TEXT,
    UNIQUE (machine_id, ts)                    -- 중복 방어 : machine_id + ts 조합은 한 번만 허용할 것
);


CREATE INDEX IF NOT EXISTS ix_sensor_ts      ON sensor_raw (ts);
CREATE INDEX IF NOT EXISTS ix_sensor_machine ON sensor_raw (machine_id, ts);

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
    "collected_at",
]

# DB에 UNIQUE로 제약을 걸고 INSERT OR IGNORE로 넣음
# 스키마로 막으면 DB가 물리적으로 막아주기 때문에 어떤 경로로 중복이 들어오든 안전하게 방어 가능


# SQLite DB에 연결
def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)  # 위에 작성한 SCHEMA 안의 SQL 실행
    return con  # 만들어진 DB 연결 객체 반환


# DataFrame의 센서 데이터를 DB에 집어넣기
# DataFrame을 DB에 넣되 (machine_id, ts) 중복 무시, 새로 인서트 된 행 수 계산, 중복 행 수 계산
def upsert(con: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    # (insert된 행 수, 중복으로 건너뛴 행 수)를 돌려줌 -> 정수 2개를 튜플로 반환
    df = df.reindex(columns=COLUMNS)
    before = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    sql = (
        f"INSERT OR IGNORE INTO sensor_raw ({','.join(COLUMNS)}) "
        f"VALUES ({','.join('?' * len(COLUMNS))})"  # 실제 값이 들어갈 자리 '?'로 표시
    )
    con.executemany(
        sql, df.where(pd.notna(df), None).itertuples(index=False, name=None)
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    inserted = after - before
    return inserted, len(df) - inserted


# pandas의 NaN을 sqlite3에 그대로 넣으면 'NULL'이 아니라 문자열의 nan이나 부동소수 NaN으로 들어가는 상황 발생
# 이를 방지하기 위해  df.where(pd.notna(df), None)을 사용해 명시적으로 NaN을 None으로 바꿔주는 작업을 수행함
# 이 작업을 거치면 최종적으로 SQL NULL이 됨


# 해당 회차의 수집 작업에서 생긴 이벤트 기록
def log_run(con, window_start, window_end, received, inserted, skipped, note=""):
    con.execute(
        "INSERT INTO collect_log (run_at, window_start, window_end,"
        " rows_received, rows_inserted, rows_skipped, note)"
        " VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (str(window_start), str(window_end), received, inserted, skipped, note),
    )
    con.commit()


# DB에 있던 센서 데이터 전체를 다시 DataFrame으로 받을 것
def read_all(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sensor_raw ORDER BY ts, machine_id", con)
