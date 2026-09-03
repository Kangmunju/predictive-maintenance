"""
CSV 이력 → SQLite 재생성
========================
data/history/*.csv 만 있으면 언제든 DB를 다시 만들 수 있습니다.
"DB는 산출물이고 CSV가 원본"이라는 구조입니다.

    python src/build_db.py
"""

# collector가 만들어둔 일별 CSV들을 읽고 SQLite DB를 처음부터 다시 만들기 위해 작성한 파일
# 앞서 정한 규칙 그대로 DB 자체를 원본으로 보지 않고 CSV를 원본으로 볼 것

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db as dbmod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"


def main() -> int:
    files = sorted(
        HIST.glob("*.csv")
    )  # 우선 data/history 폴더 안에서 확장자가 .csv 파일이면 전부 찾아올 것
    if not files:
        print(f"[WARN] {HIST} 에 CSV가 없습니다. 먼저 collector.py를 돌리세요.")
        return 1

    dbp = dbmod.DB_PATH
    if (
        dbp.exists()
    ):  # 현재까지 이미 collector를 여러번 실행한 상태이므로 sensors.db 존재함
        dbp.unlink()
        print("기존 DB 삭제 후 재생성합니다.")
    # CSV만 가지고 DB를 처음부터 다시 만들기 위해 기존 sensors.db 파일을 삭제
    # CSV만 있으면 DB를 언제든 똑같이 재생성할 수 있도록 재현 가능 파이프라인을 구현하고자 함

    con = dbmod.connect(dbp)
    # db.py에서 작성한 connect()함수 호출
    # 함수 내에 con = sqlite3.connect(path)를 작성하여
    # 기존 DB에만 연결되는 것이 아니라 해당 위치에 DB가 없으면 새 DB 파일을 만들어 연결되도록 함
    # 따라서 unlink()로 삭제했어도 error가 발생하지 않음

    total_in = total_ins = 0
    for f in files:
        df = pd.read_csv(f)
        ins, skip = dbmod.upsert(con, df)
        total_in += len(df)  # CSV에서 읽은 행 전체
        total_ins += ins  # 실제 DB에 새로 저장된 행 전체
        print(f"  {f.name:<20} 읽음 {len(df):>6,} / 신규 {ins:>6,} / 중복 {skip:>5,}")
    # CSV의 내용이 아닌 CSV 파일 자체의 경로(f)로 반복문이 동작하도록 함
    # 중복이 하나도 없는 경우 total_in과 total_ins가 동일한 결과를 리턴
    # CSV끼리 기준으로 잡았던 (machine_id, ts)가 중복인 데이터가 있는 경우 total_ins가 작아지도록 작성
    n = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    con.close()

    print(f"\nCSV {len(files)}개 / 읽은 행 {total_in:,} / DB {n:,}행")
    print(f"차이 {total_in - n:,}행은 (machine_id, ts) 중복입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------
# total_ins = CSV를 넣으며 파이썬 내에서 직접 누적해서 계산한 값 (신규 저장 행 수)
# n = 모든 작업이 끝난 후 실제 SQLite DB를 조회해서 확인한 값 (실제 총 행 수)
# 기존 DB를 삭제하고 새로 만드는 과정을 거쳤으므로 정상 처리되었다면 둘이 같은 값이 나오도록 구현

# [CSV -> DB 재생성 테스트]
# collect.py 테스트 과정에서 생성된 2024-03-01.csv가 남아있어 data/history 폴더에 총 15개의 CSV가 존재함을 확인
# 14일(2주)치 데이터를 기준으로 하여 DB 재생성 결과를 확인하고자 했던 처음의 의도를 따라가기 위해
# 2024-03-01.csv를 프로젝트 폴더 밖으로 임시로 이동시켜두었음

# PowerShell에서 Move-Item data\history\2024-03-01.csv ..\2024-03-01.csv 실행
# 이후 data/history에 14개의 CSV만 남은 것을 확인
# python src/build_db.py 실행
# 실행 결과
# CSV 14개 / 읽은 행 55,591 / DB 55,591행
# 차이 0행은 (machine_id, ts) 중복
#
# 기존 DB를 삭제한 뒤 14일치 CSV만으로 동일한 데이터를 SQLite DB에 다시 생성할 수 있음을 확인

# 이하 build_db.py파일 전체 출력 결과
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
# 차이 0행은 (machine_id, ts) 중복입니다.
