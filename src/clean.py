"""
정제 파이프라인
===============
"오염 주입"의 역순으로 벗겨냅니다. 순서가 중요합니다.

  1) 타입 강제        문자열로 온 숫자·시각을 제자리로
  2) 타임스탬프 정렬  초 단위 흔들림을 분에 스냅
  3) 중복 제거        (machine_id, ts) 기준
  4) 단위 통일        섭씨↔켈빈, m/s²↔mm/s
  5) 물리 범위 검사   불가능한 값을 NaN으로 (지우지 않음)
  6) 스파이크 탐지    Hampel 필터 — ★ 지우지 말고 플래그만
  7) 결측 보간        짧은 구간만. 긴 끊김은 그대로 남긴다
  8) 드리프트 보정    다른 설비를 기준으로 밀린 양을 추정
  9) 시간축 재색인    빠진 분을 명시적으로 드러낸다

★ 모든 단계는 StepLog에 행 수를 남깁니다.
  "원본 대비 최종 건수 차이를 설명할 수 있나?"에 답하기 위해서입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
]

# 물리적으로 가능한 범위 (설비 스펙 + 상식)
PHYS_RANGE = {
    "air_temp_k": (270.0, 340.0),  # -3 ~ 67 도
    "process_temp_k": (280.0, 360.0),
    "rot_speed_rpm": (500.0, 4000.0),
    "torque_nm": (1.0, 100.0),
    "tool_wear_min": (0.0, 400.0),
    "vibration_mms": (0.05, 30.0),  # ISO 10816: 11 mm/s 초과면 위험
    "current_a": (0.1, 40.0),
    "humidity_pct": (0.0, 100.0),
}
# 데이터를 오염시키기 위해 일부 센서값들을 비정상적으로 튀게 만들어두었음
# 설비에서 물리적으로 가능한 범위를 미리 설정
# 명백히 불가능한 값만 처리하고 행은 유지하도록 설계


# 각 단계를 밟을 때마다 행 수를 기록
class StepLog:
    def __init__(self):
        self.rows = []

    def __call__(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        prev = self.rows[-1][1] if self.rows else len(df)
        self.rows.append((name, len(df), len(df) - prev))
        return df

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["단계", "행수", "증감"])


# ----------------------------------------------------------------------
# 1~3. 타입 · 중복 · 타임스탬프
# ----------------------------------------------------------------------


# 문자열로 입력된 데이터는 이후 연산 및 분석 과정에서 오류가 발생할 수 있음
# 날짜·센서값 등 각 컬럼을 적절한 데이터 타입으로 변환
# ts(측정 시각)와 machine_id(설비 식별자)를 데이터 식별을 위한 필수 정보로 설정
# 두 값 중 하나라도 누락된 경우 측정 시점 또는 대상 설비를 특정할 수 없으므로 해당 행 제거
# 그 외 센서값의 결측은 다른 정상 센서값을 보존하기 위해 행을 삭제하지 않도록 설계
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for c in SENSOR_COLS + ["machine_failure"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts", "machine_id"])


# 초 단위로 흔들린 타임스탬프를 분 단위로 맞춰 측정 시각을 정규화
def snap_timestamp(df: pd.DataFrame, freq: str = "min") -> pd.DataFrame:
    df = df.copy()
    df["ts"] = df["ts"].dt.round(freq)
    return df


# (machine_id, ts)를 기준으로 동일 설비·동일 시각의 데이터를 중복 측정 건으로 판단
# 중복 발생 시 collected_at 기준으로 가장 나중에 수집된 데이터를 신뢰하여 유지
def drop_dups(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("collected_at")
        .drop_duplicates(subset=["machine_id", "ts"], keep="last")
        .sort_values(["machine_id", "ts"])
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------
# 4. 단위 통일
# ----------------------------------------------------------------------


# 켈빈 온도 컬럼에 혼입된 섭씨값 탐지 및 단위 정상화
# 물리적 타당성을 기준으로 섭씨 혼입 여부 판정
# 임데이터 분포가 아닌 도메인 지식에 기반한 임계값을 사용
def detect_and_fix_temp_unit(df: pd.DataFrame, cols=("air_temp_k", "process_temp_k")):
    df = df.copy()
    report = {}
    for c in cols:
        if c not in df.columns:
            continue
        mask = df[c].notna() & (df[c] < 200)
        report[c] = int(mask.sum())
        df.loc[mask, c] = df.loc[mask, c] + 273.15
    return df, report


# 진동 센서에 혼입된 m/s² 단위값을 탐지하고 vibration_mms 단위로 정상화
# 진동값은 온도와 달리 특정 값만으로 단위 오류를 확정하기 어려움
# 고장 설비에서는 높은 진동값도 실제로 발생할 수 있으므로 설비별 중앙값을 기준으로 판정
# 각 설비의 중앙값보다 ratio배 이상 큰 값을 단위 오류로 추정하여 변환
# 실제 환경에서는 메타데이터(태그 단위표)가 있다면 해당 정보를 우선적으로 사용하는 것이 적절함
def detect_vibration_unit(
    df: pd.DataFrame, col="vibration_mms", factor=9.81, ratio=4.0
):
    df = df.copy()
    med = df.groupby("machine_id")[col].transform("median")
    # 설비별 진동값의 중앙값 계산
    mask = df[col].notna() & (df[col] > med * ratio)
    # 결측값을 제외하고 설비별 중앙값의 ratio배를 초과하는 값 감지
    df.loc[mask, col] = df.loc[mask, col] / factor
    # 탐지된 값의 단위를 vibration_mms 기준으로 변환
    return df, int(mask.sum())
    # 진동 단위를 수정한 데이터프레임, 진동 단위 오류로 판단해서 수정한 데이터 개수 리턴


# ----------------------------------------------------------------------
# 5. 물리 범위 검사
# ----------------------------------------------------------------------


# 센서별 물리적 정상 범위를 기준으로 비정상값 탐지
# 해당 시각의 다른 정상 센서값을 잃는 것을 방지하기 위해
# 범위를 벗어난 값만 NaN으로 처리하고 행 자체는 유지
def range_check(df: pd.DataFrame, rng: dict | None = None):
    rng = rng or PHYS_RANGE
    # 별도 범위가 없으면 미리 정의한 PHYS_RANGE 사용
    df = df.copy()
    report = {}
    for c, (lo, hi) in rng.items():
        if c not in df.columns:
            continue
        bad = df[c].notna() & ~df[c].between(lo, hi)
        report[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    return df, report


# ----------------------------------------------------------------------
# 6. 스파이크 탐지 (Hampel)
# ----------------------------------------------------------------------
# 이상치를 삭제하지 않고 플래그로 기록하여 후속 분석 및 모델링 단계에서 활용 가능하도록 설계


# 센서값의 급격한 튐(Spike)을 탐지하여 플래그로 기록
# 이상치를 삭제하지 않고 표시하여 후속 분석 및 모델링 단계에서 활용
# Hampel Filter를 적용하여 주변 데이터의 일반적인 변동 범위를 크게 벗어난 값을 탐지
def hampel_flag(s: pd.Series, window: int = 11, n_sigma: float = 5.0) -> pd.Series:
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    sigma = sigma.replace(0, np.nan)
    return ((s - med).abs() > n_sigma * sigma).fillna(False)


# 이동 구간의 중앙값을 계산해 주변 센서값의 대표적인 수준을 med로 설정
# MAD(중앙절대편차)로 센서값이 주변 중앙값에서 일반적으로 얼마나 벗어나는지 계산
# 표준편차보다 극단값의 영향을 적게 받는 MAD를 사용하여 스파이크 탐지 기준이 흔들리지 않도록 함


# 설비별 센서값의 스파이크를 탐지하여 플래그 컬럼으로 기록
# 센서값의 급격한 변화는 실제 설비 이상 신호일 가능성이 있으므로 이상값을 임의로 삭제하지 않음
# 원본 센서값은 유지하고 스파이크 여부만 표시하여 후속 분석 및 모델링에 활용
def flag_spikes(df: pd.DataFrame, cols=None, window=11, n_sigma=5.0):
    cols = cols or SENSOR_COLS
    # 별도 col 지정이 없으면 위에서 만들어둔 8개 센서 컬럼 전부 검사
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[f"spike_{c}"] = df.groupby("machine_id")[c].transform(
            lambda s: hampel_flag(s, window, n_sigma)
        )
    spike_cols = [f"spike_{c}" for c in cols if f"spike_{c}" in df.columns]
    # cols의 센서들을 순회하면서 실제로 생성된 spike_센서명 컬럼만 골라 spike_cols 리스트로 생성
    df["spike_any"] = df[spike_cols].any(axis=1)
    df["spike_count"] = df[spike_cols].sum(axis=1)
    return df


# ----------------------------------------------------------------------
# 7. 결측 보간
# ----------------------------------------------------------------------


# 센서 데이터의 짧은 결측 구간을 설비별 시간 흐름에 따라 선형 보간
# 긴 결측 구간까지 보간하면 실제로 측정되지 않은 구간을 임의의 값으로 채우게 되므로
# 보간 가능한 결측 개수를 제한하여 실제 데이터의 흐름이 왜곡되는 것을 방지
# 센서별 실제 보간 개수를 기록하여 정제 결과를 확인
def interpolate_short_gaps(df: pd.DataFrame, cols=None, max_gap: int = 5):
    cols = cols or SENSOR_COLS
    # 별도 col 지정이 없으면 위에서 만들어둔 8개 센서 컬럼 전부 검사
    df = df.sort_values(["machine_id", "ts"]).copy()
    filled = {}
    for c in cols:
        if c not in df.columns:
            continue
        before = df[c].isna().sum()
        df[c] = df.groupby("machine_id")[c].transform(
            lambda s: s.interpolate(
                method="linear", limit=max_gap, limit_direction="both"
            )
        )
        filled[c] = int(before - df[c].isna().sum())
    return df, filled


# ----------------------------------------------------------------------
# 8. 드리프트 보정
# ----------------------------------------------------------------------


# 공정 온도와 기준 온도의 차이를 이용해 설비별 센서 드리프트를 추정
# 설비별 일일 중앙값을 전체 설비의 일일 중앙값과 비교하여 개별 설비의 변화만 분리
# 전체 설비가 함께 변하는 공정 변화와 특정 설비에서만 발생하는 센서 드리프트를 구분
def estimate_drift(df: pd.DataFrame, col="process_temp_k", ref="air_temp_k"):
    d = df.dropna(subset=[col, ref]).copy()
    d["diff"] = d[col] - d[ref]
    d["day"] = (d["ts"] - d["ts"].min()).dt.total_seconds() / 86400.0
    daily = (
        d.groupby(["machine_id", d["day"].astype(int)])["diff"]
        .median()
        .rename("v")
        .reset_index()
        .rename(columns={"day": "d"})
    )
    # v : 설비별·일별 온도 차이의 중앙값을 계산하여 하루의 대표값으로 사용
    fleet = daily.groupby("d")["v"].median().rename("fleet")
    # fleet : 같은 날 전체 설비의 중앙값을 계산하여 비교 기준으로 사용
    daily = daily.join(fleet, on="d")
    daily["resid"] = daily["v"] - daily["fleet"]
    # resid = v - fleet : 해당 설비가 전체 설비 기준에서 얼마나 벗어나는지 계산

    out = {}  # 설비별로 계산한 드리프트 기울기(slope)를 out에 딕셔너리 형태로 저장
    for m, g in daily.groupby("machine_id"):
        if len(g) < 3:
            out[m] = 0.0
            continue
        slope = np.polyfit(g["d"], g["resid"], 1)[0]
        # slope : 시간에 따른 차이의 기울기를 계산하여 설비별 드리프트 정도 추정
        out[m] = float(slope)
    return out, daily


# 추정된 드리프트 기울기를 기준으로 실제 센서값을 보정
# 작은 변화까지 센서 이상으로 판단하지 않도록 임계값 이상의 드리프트만 보정
# 시간에 따라 누적된 드리프트 양을 계산하여 해당 설비의 센서값에서 제거
def correct_drift(
    df: pd.DataFrame, slopes: dict, col="process_temp_k", min_slope: float = 0.05
):
    df = df.copy()
    t0 = df["ts"].min()
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0
    # 최초 측정 시점을 기준으로 각 데이터의 경과 일수 계산
    applied = {}
    for m, s in slopes.items():
        if abs(s) < min_slope:
            applied[m] = 0.0
            continue
        mask = df["machine_id"] == m
        df.loc[mask, col] = df.loc[mask, col] - s * days[mask]
        applied[m] = s
    return df, applied


# 기울기의 절댓값이 임계값보다 작으면 보정하지 않음
# 시간에 따라 누적된 드리프트를 계산하여 해당 설비의 센서값에서 제거


# ----------------------------------------------------------------------
# 9. 시간축 재색인
# ----------------------------------------------------------------------


# 설비별로 빠진 시간 구간을 찾아 NaN 행으로 명시
# 실제 측정값이 없던 시점을 is_gap으로 표시하여 결측 시간 자체를 정보로 유지
# 설비별 시간축을 일정한 주기로 맞춘 뒤 다시 하나의 데이터로 결합
def reindex_time(df: pd.DataFrame, freq: str = "min") -> pd.DataFrame:
    parts = []
    for m, g in df.groupby("machine_id"):
        g = g.set_index("ts").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        # 해당 설비의 처음부터 마지막 시점까지 일정한 간격의 전체 시간축 생성
        g2 = g.reindex(full)
        # 실제로 존재하지 않았던 시간은 NaN 행으로 추가
        g2["is_gap"] = g2["machine_id"].isna()
        # 새로 생성된 행을 실제 결측 시간으로 표시
        g2["machine_id"] = m
        g2["type"] = g["type"].iloc[0] if "type" in g.columns else None
        g2.index.name = "ts"
        parts.append(g2.reset_index())
    return pd.concat(parts, ignore_index=True).sort_values(["ts", "machine_id"])


# ----------------------------------------------------------------------
# 전체 파이프라인
# ----------------------------------------------------------------------
def run_pipeline(raw: pd.DataFrame, verbose: bool = True):
    log = StepLog()
    rep = {}

    df = log("0. 원본 수신", raw.copy())
    df = log("1. 타입 강제", coerce_types(df))
    df = log("2. 타임스탬프 스냅", snap_timestamp(df))
    df = log("3. 중복 제거", drop_dups(df))

    df, rep["temp_unit"] = detect_and_fix_temp_unit(df)
    df = log("4a. 온도 단위 통일", df)
    df, rep["vib_unit"] = detect_vibration_unit(df)
    df = log("4b. 진동 단위 통일", df)

    df, rep["range"] = range_check(df)
    df = log("5. 물리범위 → NaN", df)

    df = flag_spikes(df)
    df = log("6. 스파이크 플래그", df)

    df, rep["filled"] = interpolate_short_gaps(df)
    df = log("7. 짧은 결측 보간", df)

    slopes, rep["drift_daily"] = estimate_drift(df)
    rep["drift_slopes"] = slopes
    df, rep["drift_applied"] = correct_drift(df, slopes)
    df = log("8. 드리프트 보정", df)

    df = reindex_time(df)
    df = log("9. 시간축 재색인", df)

    if verbose:
        print(log.frame().to_string(index=False))
    return df, log, rep


# ----------------------------------------------------------------------
# 정제는 오염이 발생한 과정을 고려해 순서대로 수행
# 특히 단위 통일 후 물리 범위를 검사하여 정상 데이터를 이상값으로 오인하지 않도록 함
#
# 한 센서값에 문제가 있어도 같은 시각의 다른 센서값은 정상일 수 있으므로
# 물리적으로 불가능한 값만 해당 셀을 NaN으로 처리하고 행 자체는 최대한 유지함
#
# 스파이크는 실제 설비 이상 신호일 수 있으므로 삭제하지 않고 플래그로 기록함.
# Hampel Filter는 주변 값의 중앙값과 MAD를 이용하여 극단값의 영향을 적게 받는 기준으로 스파이크를 탐지
# 현재 Hampel Filter는 center=True로 미래 시점까지 함께 사용하므로 사후 전처리에 적합하며,
# 실시간 탐지에서는 미래 데이터가 포함되지 않도록 처리 방식을 변경해야 함.
#
# 보간은 앞뒤 실제 측정값을 이용해 중간의 누락값을 추정하는 방법
# 시간 흐름이 있는 센서 데이터의 짧은 결측은 앞뒤 변화 흐름을 반영할 수 있도록 선형보간을 사용
# 긴 결측까지 보간하면 실제로 측정되지 않은 구간을 만들어낼 수 있으므로 보간 범위를 제한
#
# 드리프트는 스파이크처럼 순간적으로 튀는 값이 아니라 시간이 지나면서 센서값이 서서히 밀리는 현상
# 공정온도와 공기온도의 차이를 사용해 외부 온도 변화를 줄이고
# 전체 설비의 일별 중앙값과 비교하여 공정 전체의 변화와 특정 설비의 변화를 구분
# 잔차의 시간에 따른 기울기(slope)로 드리프트 정도를 추정
# 작은 변화까지 임의로 수정하지 않도록 일정 임계값 이상의 드리프트만 보정
#
# 원본 데이터에서 행 자체가 사라진 시간은 단순 isna()만으로 확인할 수 없음
# 일정한 시간축으로 재색인하여 빠진 시점을 NaN 행으로 명시
# 새로 생성된 행은 is_gap으로 표시하여 실제 측정값이 없었던 시간이라는 정보를 유지
#
# StepLog로 각 정제 단계의 행 수와 증감을 기록
# 원본에서 최종 데이터까지 어떤 단계에서 행 수가 변했는지 확인할 수 있도록 함
