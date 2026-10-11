"""
주기 수집기
===========
GitHub Actions가 하루 한 번 이 파일을 실행합니다.

    python src/collector.py --minutes 1440

동작
  1) 센서 소스에서 최근 N분 구간을 받아온다
  2) data/history/YYYY-MM-DD.csv 로 원본 그대로 저장 (감사 추적)
  3) SQLite에 UPSERT (중복은 DB가 막음)
  4) collect_log에 수집 이력을 남긴다

★ 왜 CSV와 SQLite를 둘 다 쓰나
   - CSV: git에 커밋해도 diff가 보인다. "언제 뭐가 들어왔는지" 추적 가능
   - SQLite: 조회·조인이 편하다. 하지만 바이너리라 git에 넣으면 diff가 안 보인다
   그래서 CSV만 커밋하고, DB는 CSV로부터 언제든 재생성합니다(build_db.py).
   이 구조를 면접에서 설명하면 "재현 가능한 파이프라인"을 아는 사람으로 보입니다.
"""

# 센서 데이터 가져오기 -> CSV 저장 -> SQLite 저장 -> 수집 기록 남기기

from __future__ import annotations

import argparse  # 터미널에서 프로그램을 실행하기 위해 값을 전달받거나 입력한 옵션을 받아줘야 함
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import db as dbmod  # db.py를 dbmod로 칭하며 사용함 # noqa: E402
from simulator import sample_window  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"

# 자동 공백 복구가 지나치게 오래된 데이터를 한 번에 생성하지 않도록 제한
MAX_RECOVERY_MINUTES = 7 * 24 * 60


# ----------------------------------------------------------------------
# 센서 소스 호출
# ----------------------------------------------------------------------
# 포트폴리오 작성용 함수가 아닌 실무에서는 이 함수를 REST API 호출 용도로 사용
# 현재 폴더에는 실제 센서가 없기 때문에 앞서 작성한 시뮬레이터를 호출
# r = requests.get(API, params={...}, timeout=10)
# r.raise_for_status()
# return pd.DataFrame(r.json()["items"])
def fetch(minutes: int, end: str | None = None) -> pd.DataFrame:
    return sample_window(n_minutes=minutes, end=end)


# simulator.py에 작성한 sample_window() 함수로 지정한 시간 구간의 센서 데이터를 생성
# 수정된 sample_window()는 관측 시각과 고정 seed를 기준으로 데이터를 생성함
# 같은 시간 구간을 다시 수집하면 동일한 센서 데이터가 생성됨
# 이를 통해 같은 데이터를 반복 수집해도 중복 저장되지 않는지 확인할 수 있도록 프로그램을 구성
# 단, collected_at은 실제 수집 실행 시각이므로 재실행할 때 달라질 수 있음


# ----------------------------------------------------------------------
# 수집 구간 끝 시각을 UTC 기준으로 정리
# ----------------------------------------------------------------------
def normalize_end(end: str | None) -> pd.Timestamp:
    if end is None:
        return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")

    ts = pd.Timestamp(end)

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts.floor("min")


# ----------------------------------------------------------------------
# 마지막으로 저장된 신규 관측 시각 확인
# ----------------------------------------------------------------------
def last_observed_at(con) -> pd.Timestamp | None:
    row = con.execute(
        """
        SELECT MAX(observed_at)
        FROM sensor_raw
        WHERE observed_at IS NOT NULL
        """
    ).fetchone()

    if row is None or row[0] is None:
        return None

    return pd.Timestamp(row[0])


# ----------------------------------------------------------------------
# 수집 공백 복구 구간 계산
# ----------------------------------------------------------------------
def resolve_minutes(
    con,
    requested_minutes: int,
    end: pd.Timestamp,
    recover: bool,
) -> int:
    if requested_minutes <= 0:
        raise ValueError("--minutes는 양수여야 합니다.")

    if not recover:
        return requested_minutes

    last = last_observed_at(con)

    if last is None:
        # 신규 관측 데이터가 없는 최초 실행에서는 요청 구간만 수집
        return requested_minutes

    # 마지막 관측 시각부터 이번 실행 시각까지의 경과 분
    elapsed = int((end - last).total_seconds() // 60)

    if elapsed <= 0:
        return requested_minutes

    # sample_window()는 종료 시각을 포함하지 않음
    # 마지막 관측 시각 다음 분부터 종료 직전까지 수집하려면
    # 경과 시간인 elapsed분만큼 수집하면 됨
    needed = elapsed

    # 기존 요청 구간보다 공백이 길면 복구 범위를 확장
    minutes = max(requested_minutes, needed)

    # 지나치게 긴 공백은 한 번에 처리하지 않음
    return min(minutes, MAX_RECOVERY_MINUTES)


# ----------------------------------------------------------------------
# 날짜별 CSV 저장
# ----------------------------------------------------------------------
def save_daily_csv(raw: pd.DataFrame, hist: Path) -> list[Path]:
    """
    observed_at의 UTC 날짜를 기준으로 CSV를 분리함.

    같은 관측값이 다시 들어오면 기존 CSV 행을 유지하고,
    신규 관측값만 추가함.
    """
    hist.mkdir(parents=True, exist_ok=True)

    data = raw.copy()

    if "observed_at" not in data.columns:
        raise ValueError("수집 데이터에 observed_at 컬럼이 없습니다.")

    data["_date"] = pd.to_datetime(data["observed_at"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )

    paths = []

    for date, group in data.groupby("_date", sort=True):
        csv_path = hist / f"{date}.csv"

        new_rows = group.drop(columns=["_date"]).copy()

        if csv_path.exists():
            old = pd.read_csv(csv_path)

            # 기존 데이터에 observed_at이 없는 경우
            # 과거 수집 데이터의 ts를 임의로 복원하지 않음
            if "observed_at" not in old.columns:
                old["observed_at"] = pd.NA

            # 기존 CSV의 컬럼을 먼저 유지하고 신규 컬럼을 추가
            columns = list(old.columns)

            for col in new_rows.columns:
                if col not in columns:
                    columns.append(col)

            old = old.reindex(columns=columns)
            new_rows = new_rows.reindex(columns=columns)

            combined = pd.concat(
                [old, new_rows],
                ignore_index=True,
            )
        else:
            combined = new_rows

        # 기존 데이터는 observed_at이 없을 수 있으므로
        # 신규 데이터와 기존 데이터의 중복 기준을 구분
        legacy = combined[combined["observed_at"].isna()].copy()
        current = combined[combined["observed_at"].notna()].copy()

        # 과거 데이터는 기존 방식으로 중복 제거
        legacy = legacy.drop_duplicates(
            subset=["machine_id", "ts"],
            keep="first",
        )

        # 신규 데이터는 실제 관측 기준 시각으로 중복 제거
        current = current.drop_duplicates(
            subset=["machine_id", "observed_at"],
            keep="first",
        )

        combined = pd.concat(
            [legacy, current],
            ignore_index=True,
        )

        # 관측 시각 순으로 정렬하되 원본 값은 변경하지 않음
        combined["_sort_ts"] = pd.to_datetime(
            combined["observed_at"].fillna(combined["ts"]),
            errors="coerce",
        )

        combined = combined.sort_values(
            ["_sort_ts", "machine_id"],
            kind="stable",
        ).drop(columns=["_sort_ts"])

        combined.to_csv(csv_path, index=False)
        paths.append(csv_path)

    return paths


# ----------------------------------------------------------------------
# 실제 수집 프로그램
# ----------------------------------------------------------------------
# 가져온 데이터를 CSV와 DB에 실제 저장
def main() -> int:
    ap = argparse.ArgumentParser()  # 터미널에서 들어오는 옵션들을 관리할 객체 생성

    ap.add_argument(
        "--minutes",
        type=int,
        default=1440,
        help="수집할 구간 길이(분)",
    )  # default=1440minutes(최근 하루치 데이터)

    ap.add_argument(
        "--end",
        default=None,
        help="구간 끝 시각(기본: 지금, UTC 기준)",
    )  # 수집 구간의 끝 시각 설정

    ap.add_argument("--db", default=str(dbmod.DB_PATH))
    ap.add_argument("--hist", default=str(HIST), help="일별 CSV 저장 폴더")

    ap.add_argument(
        "--recover",
        action="store_true",
        help="마지막 관측 시각 이후의 수집 공백 자동 복구",
    )

    # 똑같은 수집 데이터를 서로 다른 형태(CSV 파일, SQLite DB)로 두 번 보관
    # CSV는 원본 기록 보관 용으로 사용할 것
    # SQLite는 주로 그 데이터를 조회하고 사용할 것

    args = ap.parse_args()

    hist = Path(args.hist)
    hist.mkdir(parents=True, exist_ok=True)

    try:
        end = normalize_end(args.end)

        # DB에 저장된 마지막 관측 시각을 기준으로 복구 구간 계산
        con = dbmod.connect(args.db)

        try:
            minutes = resolve_minutes(
                con,
                args.minutes,
                end,
                args.recover,
            )
        finally:
            con.close()

        if args.recover and minutes != args.minutes:
            print(f"[INFO] 수집 공백 복구: {args.minutes:,}분 -> {minutes:,}분")

        # 실제로 센서 데이터를 가져오는 부분
        raw = fetch(
            minutes,
            end.strftime("%Y-%m-%d %H:%M:%S"),
        )

    except Exception as e:  # 수집 실패 시 오류를 출력하고 종료
        print(f"[ERROR] 수집 실패: {type(e).__name__}: {e}")
        return 1

    if raw.empty:
        print("[WARN] 받은 데이터가 0건입니다. 종료합니다.")
        return 0

    # fetch() 자체는 오류 없이 성공했더라도
    # 결과가 비어 있는 경우가 존재할 수 있음
    # 데이터가 없는 경우라면 이하 코드 진행하지 않음

    # ts에는 센서 타임스탬프 흔들림이 적용될 수 있으므로
    # 수집 구간은 observed_at을 기준으로 계산함
    w_start = raw["observed_at"].min()
    w_end = raw["observed_at"].max()

    # 날짜별 CSV 저장
    try:
        csv_paths = save_daily_csv(raw, hist)
    except Exception as e:
        print(f"[ERROR] CSV 저장 실패: {type(e).__name__}: {e}")
        return 1

    # SQLite에는 이번 실행에서 실제로 수집한 raw만 적재
    # 기존 CSV의 모든 행을 다시 넣지 않도록 분리함
    try:
        con = dbmod.connect(args.db)

        try:
            inserted, skipped = dbmod.upsert(con, raw)

            note = (
                f"csv={','.join(path.name for path in csv_paths)}"
                f"; recover={args.recover}"
                f"; minutes={minutes}"
            )

            dbmod.log_run(
                con,
                w_start,
                w_end,
                len(raw),
                inserted,
                skipped,
                note=note,
            )

            total = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]

        finally:
            con.close()

    except Exception as e:
        print(f"[ERROR] DB 저장 실패: {type(e).__name__}: {e}")
        return 1

    print(f"[OK] window {w_start} ~ {w_end}")
    print(f"     받은 행 {len(raw):,} / DB 신규 {inserted:,} / 중복 스킵 {skipped:,}")

    for csv_path in csv_paths:
        print(f"     CSV  {csv_path}")

    print(f"     DB 누적 {total:,}행")

    # 실행 1) 2시간짜리 구간
    # [OK] window 2024-03-01 07:00:00 ~ 2024-03-01 08:31:10
    #  받은 행 275 / DB 신규 275 / 중복 스킵 0
    #  CSV  C:\Users\swrkd\Desktop\predictive-maintenance\data\history\2024-03-01.csv
    #  DB 누적 275행
    # 실행 2) 2시간짜리 구간
    # 똑같은 수집 작업을 여러 번 실행했을 때 중복 데이터가 DB에 또 들어가는 지 확인하기 위해 동일 조건으로 재검
    # [OK] window 2024-03-01 07:00:00 ~ 2024-03-01 08:31:10
    #  받은 행 275 / DB 신규 0 / 중복 스킵 275
    #  CSV  C:\Users\swrkd\Desktop\predictive-maintenance\data\history\2024-03-01.csv
    #  DB 누적 275행
    # 중복 방지가 제대로 작동함을 확인하였으므로 검사를 계속 진행하지 않고 멈춤

    return 0

    # [14일치 수집 테스트]
    # collector.py가 여러 날짜의 데이터를 연속으로 수집하고 저장하는지 확인하기 위해
    # 2024-01-01 ~ 2024-01-14까지 총 14일치 데이터를 수집함
    # 실행 1) 2024-01-01 하루치를 테스트 용도로 실행
    # python src/collector.py --minutes 1440 --end "2024-01-02 00:00:00"
    # [OK] window 2024-01-01 00:00:00 ~ 2024-01-01 23:59:00
    # 받은 행 3,903 / DB 신규 3,903 / 중복 스킵 0
    # DB 누적 4,178행
    # 실행 2) PowerShell의 for문을 사용하여 남은 13일치를 연속 실행
    # for ($i = 1; $i -le 13; $i++) {
    #     $d = (Get-Date "2024-01-02").AddDays($i).ToString("yyyy-MM-dd HH:mm:ss")
    #     python src/collector.py --minutes 1440 --end "$d"
    # }
    # 2024-01-02 ~ 2024-01-14까지 날짜별 CSV가 생성되는 것을 확인함
    # 14일치 데이터 총 55,591행 수집
    # 기존 테스트 데이터 275행을 포함하여 DB 누적 55,866행
    # 날짜별 CSV 14개 생성
    # CSV와 SQLite DB에 누적 저장할 수 있음을 확인


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------------
# CSV는 수집한 원본 데이터를 보존하고 변경 이력을 추적하기 위한 용도로 사용
# SQLite는 데이터를 효율적으로 조회하고 활용하기 위한 용도로 사용
# 원본 데이터와 활용 데이터를 목적에 따라 분리하여 관리하도록 구성
# SQLite DB는 CSV 원본 데이터를 기반으로 언제든 재생성할 수 있도록 설계

# CSV 안에서의 중복 제거와 DB에 이미 저장된 데이터와의 중복 검사가 서로 다른 단계
# 이러한 이유 때문에 CSV에서 중복을 제거하는 과정을 거쳤음에도 DB에서 '중복 스킵'건이 나오게 됨
# CSV의 drop_duplicates() -> 같은 CSV 안에서 중복 제거
# DB의 UNIQUE + INSERT OR IGNORE -> DB에 이미 저장된 데이터와 중복인지 검사
# 여러 단계로 중복을 방어하기 위하여 이렇게 구성함

# 동일한 데이터를 생성하는 역할은 sample_window()의 고정 seed가 맡도록 하였고
# 그 동일한 데이터가 DB에 또 들어가지 않도록 방지하는 역할은 db.py의 중복 방지 로직으로 구현함

# 신규 데이터는 observed_at을 기준으로 중복을 판단함
# ts는 센서 타임스탬프 흔들림이 적용될 수 있어 중복 기준으로 적합하지 않음
# collected_at은 실제 수집 실행 시각이므로 동일 구간을 재수집하면 달라질 수 있음

# =============================================================================
# [구현 핵심]
# =============================================================================

# GitHub Actions에서 실행할 수 있는 주기 수집기를 구현함
# 센서 시뮬레이터의 데이터를 CSV와 SQLite에 각각 저장하도록 구성함
# CSV는 변경 이력과 재현성을 위한 원본 보관용으로 사용함
# SQLite는 데이터 조회와 중복 방지를 위한 저장소로 사용함
# 최초 구현에서는 수집 구간의 마지막 ts 날짜를 기준으로 CSV 하나에 저장함
# 기존 CSV와 신규 데이터를 합친 후 (machine_id, ts) 기준으로 중복을 제거함
# 동일 구간 재실행 시 DB의 UNIQUE 제약으로 중복 저장을 방지함

# 수집 구간이 자정을 넘으면 다른 날짜의 데이터가 하나의 CSV에 섞이는 문제를 확인함
# observed_at의 UTC 날짜를 기준으로 CSV를 분리 저장하도록 변경함
# 기존 CSV의 행을 우선 유지해 재수집 시 collected_at이 바뀌어도 원본을 보존함
# 신규 데이터의 중복 기준을 (machine_id, observed_at)으로 변경함
# 마지막 관측 시각을 기준으로 누락된 수집 구간을 복구하는 --recover 옵션을 추가함
# 복구 범위를 최대 7일로 제한해 비정상적으로 큰 데이터 생성을 방지함
# 수집 실행마다 실제 수신 행과 DB 신규·중복 행 수를 collect_log에 기록함

# 추후 보완 사항
# 1. 7일을 초과하는 수집 공백을 여러 구간으로 나누어 복구하는 기능
# 2. CSV 저장과 SQLite 적재 중 한쪽만 실패할 때의 복구 및 재시도 처리
# 3. 자동 수집 시 실행 간격과 실제 누락 구간을 비교하는 모니터링 기능
# 4. 대용량 수집 시 CSV 파일 전체를 다시 읽지 않는 증분 저장 방식
