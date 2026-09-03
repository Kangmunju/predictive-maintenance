#


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


# 센서 소스 호출
# 포트폴리오 작성용 함수가 아닌 실무에서는 이 함수를 REST API 호출 용도로 사용
# 현재 폴더에는 실제 센서가 없기 때문에 앞서 작성한 시뮬레이터를 호출
# r = requests.get(API, params={...}, timeout=10)
# r.raise_for_status()
# return pd.DataFrame(r.json()["items"])
def fetch(minutes: int, end: str | None = None) -> pd.DataFrame:
    return sample_window(n_minutes=minutes, end=end)


# simulator.py에 작성한 sample_window() 함수로 지정한 시간 구간의 센서 데이터를 생성
# sample_window()는 시작 시각을 기준으로 seed를 생성함
# 같은 시간 구간을 다시 수집하면 동일한 센서 데이터가 생성됨
# 이를 통해 같은 데이터를 반복 수집해도 중복 저장되지 않는 지 확인할 수 있도록 프로그램을 구성


# 실제 수집 프로그램
# 가져온 데이터를 CSV와 DB에 실제 저장
def main() -> int:
    ap = argparse.ArgumentParser()  # 터미널에서 들어오는 옵션들을 관리할 객체 생성
    ap.add_argument(
        "--minutes", type=int, default=1440, help="수집할 구간 길이(분)"
    )  # default=1440minutes(최근 하루치 데이터)
    ap.add_argument(
        "--end", default=None, help="구간 끝 시각(기본: 지금)"
    )  # 수집 구간의 끝 시각 설정
    ap.add_argument("--db", default=str(dbmod.DB_PATH))
    ap.add_argument("--hist", default=str(HIST), help="일별 CSV 저장 폴더")
    # 똑같은 수집 데이터를 서로 다른 형태(CSV 파일, SQLite DB)로 두 번 보관
    # CSV는 원본 기록 보관 용으로 사용할 것
    # SQLite는 주로 그 데이터를 조회하고 사용할 것
    args = ap.parse_args()

    hist = Path(args.hist)
    hist.mkdir(parents=True, exist_ok=True)

    # 실제로 센서 데이터를 가져 오는 부분
    try:
        raw = fetch(
            args.minutes, args.end
        )  # 수집 성공했다면 최종적으로 센서 데이터가 만들어져 raw에 들어옴
    except Exception as e:  # 수집 실패해도 워크플로는 죽지 않게
        print(f"[ERROR] 수집 실패: {type(e).__name__}: {e}")
        return 1

    if raw.empty:
        print("[WARN] 받은 데이터가 0건입니다. 종료합니다.")
        return 0
    # fetch() 자체는 오류 없이 성공했더라고 결과가 비어있는 경우가 존재할 수도 있음
    # 데이터가 없는 경우라면 이하 코드 진행하지 않음

    w_start, w_end = (
        raw["ts"].min(),
        raw["ts"].max(),
    )  # 수집한 센서 데이터의 [시간]열 중 가장 이른 시간(min)과 가장 늦은 시간(max)
    tag = pd.Timestamp(w_end).strftime("%Y-%m-%d")
    csv_path = (
        hist / f"{tag}.csv"
    )  # 이후 과정을 위해 수집한 데이터를 날짜별 CSV 파일로 저장할 것

    # 같은 날 여러 번 돌아도 안전하게 설계
    # 기존 파일과 합쳐 중복 제거
    if csv_path.exists():  # 경로에 실제로 파일이 있는 지 확인해 T/F로 반환
        old = pd.read_csv(csv_path)
        raw = pd.concat([old, raw], ignore_index=True)
        # 기존 데이터와 새로 들어온 데이터를 합치기만 했기 때문에 중복 남아있음
    raw = raw.drop_duplicates(subset=["machine_id", "ts"], keep="last")
    # 이 부분에서 subset=["machine_id", "ts"]을 기준으로 중복을 제거하고 마지막 데이터만 세이브
    raw.to_csv(csv_path, index=False)

    # CSV로 저장한 raw 데이터를 SQLite에도 넣어줄 것
    con = dbmod.connect(args.db)
    inserted, skipped = dbmod.upsert(con, raw)
    # 새로 인서트된 행 수와 중복으로 인해 건너뛴 행 수 리턴
    dbmod.log_run(
        con, w_start, w_end, len(raw), inserted, skipped, note=f"csv={csv_path.name}"
    )
    # db.py에서 작성했던 log_run()을 호출
    # 해당 회차 수집 작업의 기록을 colloect_log에 남김
    # 센서 데이터 자체가 아님에 주의할 것

    total = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    con.close()

    print(f"[OK] window {w_start} ~ {w_end}")
    print(f"     받은 행 {len(raw):,} / DB 신규 {inserted:,} / 중복 스킵 {skipped:,}")
    print(f"     CSV  {csv_path}")
    print(f"     DB 누적 {total:,}행")
    # 실행결과 1)
    # [OK] window 2024-03-01 07:00:00 ~ 2024-03-01 08:31:10
    #  받은 행 275 / DB 신규 275 / 중복 스킵 0
    #  CSV  C:\Users\swrkd\Desktop\predictive-maintenance\data\history\2024-03-01.csv
    #  DB 누적 275행
    # 실행결과 2)
    # 똑같은 수집 작업을 여러 번 실행했을 때 중복 데이터가 DB에 또 들어가는 지 확인하기 위해 동일 조건으로 재검
    # [OK] window 2024-03-01 07:00:00 ~ 2024-03-01 08:31:10
    #  받은 행 275 / DB 신규 0 / 중복 스킵 275
    #  CSV  C:\Users\swrkd\Desktop\predictive-maintenance\data\history\2024-03-01.csv
    #  DB 누적 275행
    # 중복 방지가 제대로 작동함을 확인하였으므로 검사를 계속 진행하지 않고 멈춤
    return 0


# main() 내에 있는 실제 작업들을 시작
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

# 동일한 데이터를 생성하는 역할은 sample_window()의 seed가 맡도록 하였고
# 그 동일한 데이터가 DB에 또 들어가지 않도록 방지하는 역할은 db.py의 중복 방지 로직으로 구현함
