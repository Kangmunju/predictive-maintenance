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
def range_check(df: pd.DataFrame, rng: dict | None = None):
    """범위 밖 값을 NaN으로 바꿉니다. ★ 행을 지우지 않습니다.

    행을 지우면 그 시각의 다른 정상 센서값까지 함께 잃습니다.
    """
    rng = rng or PHYS_RANGE
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
def hampel_flag(s: pd.Series, window: int = 11, n_sigma: float = 5.0) -> pd.Series:
    """이동 중앙값에서 n_sigma * MAD 이상 떨어진 점을 True로 표시합니다.

    표준편차가 아니라 MAD를 쓰는 이유: 스파이크 자체가 표준편차를 부풀려서
    정작 그 스파이크를 못 잡습니다(이상치가 자기 기준을 망침).
    """
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    sigma = sigma.replace(0, np.nan)
    return ((s - med).abs() > n_sigma * sigma).fillna(False)


def flag_spikes(df: pd.DataFrame, cols=None, window=11, n_sigma=5.0):
    """★★ 플래그만 답니다. 지우지 않습니다.

    설비 이상은 '값이 튀는 것'으로 나타납니다.
    스파이크를 무조건 지우면 고장 신호를 지우게 됩니다. (5장에서 실측으로 보여드립니다)
    """
    cols = cols or SENSOR_COLS
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[f"spike_{c}"] = df.groupby("machine_id")[c].transform(
            lambda s: hampel_flag(s, window, n_sigma)
        )
    spike_cols = [f"spike_{c}" for c in cols if f"spike_{c}" in df.columns]
    df["spike_any"] = df[spike_cols].any(axis=1)
    df["spike_count"] = df[spike_cols].sum(axis=1)
    return df


# ----------------------------------------------------------------------
# 7. 결측 보간
# ----------------------------------------------------------------------
def interpolate_short_gaps(df: pd.DataFrame, cols=None, max_gap: int = 5):
    """max_gap분 이하의 짧은 구간만 시간 보간합니다.

    ★ 긴 끊김을 보간하면 '없던 데이터를 만들어내는' 것이 됩니다.
    30분 통신 두절 구간을 직선으로 채우면 모델은 그 30분을 '아주 안정적인 구간'으로
    배웁니다. 실제로는 아무 정보가 없는데도 말입니다.
    """
    cols = cols or SENSOR_COLS
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
def estimate_drift(df: pd.DataFrame, col="process_temp_k", ref="air_temp_k"):
    """설비별로 (col - ref)의 일별 중앙값이 시간에 따라 밀리는지 봅니다.

    같은 라인의 다른 설비를 기준선으로 씁니다.
    '설비 전체가 같이 오르면 공정 변화, 한 대만 오르면 센서 문제'라는 논리입니다.
    """
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
    fleet = daily.groupby("d")["v"].median().rename("fleet")
    daily = daily.join(fleet, on="d")
    daily["resid"] = daily["v"] - daily["fleet"]

    out = {}
    for m, g in daily.groupby("machine_id"):
        if len(g) < 3:
            out[m] = 0.0
            continue
        slope = np.polyfit(g["d"], g["resid"], 1)[0]
        out[m] = float(slope)
    return out, daily


def correct_drift(
    df: pd.DataFrame, slopes: dict, col="process_temp_k", min_slope: float = 0.05
):
    """추정된 기울기가 임계 이상인 설비만 보정합니다."""
    df = df.copy()
    t0 = df["ts"].min()
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0
    applied = {}
    for m, s in slopes.items():
        if abs(s) < min_slope:
            applied[m] = 0.0
            continue
        mask = df["machine_id"] == m
        df.loc[mask, col] = df.loc[mask, col] - s * days[mask]
        applied[m] = s
    return df, applied


# ----------------------------------------------------------------------
# 9. 시간축 재색인
# ----------------------------------------------------------------------
def reindex_time(df: pd.DataFrame, freq: str = "min") -> pd.DataFrame:
    """빠진 분을 NaN 행으로 명시합니다. 'is_gap' 컬럼으로 표시합니다."""
    parts = []
    for m, g in df.groupby("machine_id"):
        g = g.set_index("ts").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g2 = g.reindex(full)
        g2["is_gap"] = g2["machine_id"].isna()
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
