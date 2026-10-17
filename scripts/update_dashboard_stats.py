from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "history"
HTML = ROOT / "docs" / "index.html"


def main():

    # HISTORY 폴더에서 확장자가 .csv인 파일의 경로를 모두 찾음
    # glob("*.csv")은 CSV 파일 목록을 반함
    # sorted()를 사용해 파일명을 기준으로 정렬함
    # 데이터 수집 순서를 일정하게 유지하기 위한 것으로 CSV 내부의 행을 정렬하는 것은 아님
    files = sorted(HISTORY.glob("*.csv"))

    if not files:
        raise RuntimeError("수집된 CSV 파일이 없습니다.")

    frames = [pd.read_csv(path) for path in files]
    raw_rows = sum(len(frame) for frame in frames)

    df = pd.concat(frames, ignore_index=True)

    # Streamlit load_history()와 동일한 정제 기준
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df.drop_duplicates(subset=["machine_id", "ts"])

    days = df["ts"].dt.normalize().nunique()
    rows = len(df)
    csv_count = len(files)

    html = HTML.read_text(encoding="utf-8")

    # D.hero의 운영 일수와 데이터 행 수만 교체
    pattern = r'("hero"\s*:\s*\{\s*"days"\s*:\s*)\d+(\s*,\s*"rows"\s*:\s*)\d+'

    replacement = lambda match: f"{match.group(1)}{days}{match.group(2)}{rows}"

    html, hero_count = re.subn(
        pattern,
        replacement,
        html,
        count=1,
    )

    if hero_count != 1:
        raise RuntimeError("D.hero의 days/rows 위치를 찾지 못했습니다.")

    # 웹페이지 하단의 고정 데이터 통계 갱신
    footer_pattern = (
        r"(DATA:\s*배포 운영 CSV\s*)\d+"
        r"(개\s*·\s*원본\s*)[\d,]+"
        r"(행\s*·\s*중복 제거 후\s*)[\d,]+"
        r"(행)"
    )

    footer_replacement = lambda match: (
        f"{match.group(1)}{csv_count}"
        f"{match.group(2)}{raw_rows:,}"
        f"{match.group(3)}{rows:,}"
        f"{match.group(4)}"
    )

    html, footer_count = re.subn(
        footer_pattern,
        footer_replacement,
        html,
        count=1,
    )

    if footer_count != 1:
        raise RuntimeError("하단 데이터 통계 위치를 찾지 못했습니다.")

    HTML.write_text(html, encoding="utf-8")

    print(f"[OK] CSV 파일: {csv_count}개")
    print(f"[OK] 원본 행 수: {raw_rows:,}행")
    print(f"[OK] 중복 제거 후: {rows:,}행")
    print(f"[OK] 운영 데이터 존재 일수: {days}일")
    print("[OK] docs/index.html 운영 통계 갱신 완료")


if __name__ == "__main__":
    main()


# =============================================================================
# [구현 핵심]
# =============================================================================

# GitHub Pages의 운영 데이터 통계가 고정된 값으로 표시되는 문제를 발견함
# data/history 폴더의 CSV를 기준으로 통계를 자동 계산하는 기능을 구현하는 것으로 해결함
# glob("*.csv")과 sorted()로 CSV 파일 경로를 일정한 순서로 불러옴
# CSV 파일 개수와 원본 행 수는 데이터 정제 이전을 기준으로 계산함

# Streamlit의 load_history()와 동일한 데이터 정제 기준을 적용함
# ts를 datetime 형식으로 변환하고 유효하지 않은 타임스탬프를 제거함
# (machine_id, ts)를 기준으로 중복 행을 제거해 실제 운영 데이터 행 수를 계산함
# 운영 일수는 정제된 ts의 고유 날짜 개수로 계산함
# CSV 파일 개수나 최초 날짜와 최종 날짜의 차이를 운영 일수로 사용하지 않음
