"""
정제 파이프라인
===============

"오염 주입"의 역순으로 벗겨냅니다. 순서가 중요합니다.

1) 타입 강제        문자열로 온 숫자·시각을 제자리로
2) 타임스탬프 정렬  초 단위 흔들림을 분에 스냅
3) 중복 제거        (machine_id, ts) 기준
4) 단위 통일        섭씨↔켈빈, m/s²↔mm/s
5) 물리 범위 검사   불가능한 값을 NaN으로 (지우지 않음)
6) 스파이크 탐지    Hampel 필터 — 지우지 말고 플래그만
7) 결측 보간        짧은 구간만. 긴 끊김은 그대로 남긴다
8) 드리프트 보정    다른 설비를 기준으로 밀린 양을 추정
9) 시간축 재색인    빠진 분을 명시적으로 드러낸다

모든 단계는 StepLog에 행 수를 남깁니다.
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
    "air_temp_k": (270.0, 340.0),
    "process_temp_k": (280.0, 360.0),
    "rot_speed_rpm": (500.0, 4000.0),
    "torque_nm": (1.0, 100.0),
    "tool_wear_min": (0.0, 400.0),
    "vibration_mms": (0.05, 30.0),
    "current_a": (0.1, 40.0),
    "humidity_pct": (0.0, 100.0),
}

# 명백히 불가능한 센서값만 NaN 처리하고 행은 유지하도록 설계


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


# 날짜·센서값 등 각 컬럼을 적절한 데이터 타입으로 변환
# ts와 machine_id는 필수 식별 정보이므로 누락 시 해당 행 제거
# 그 외 센서값의 결측은 행을 삭제하지 않고 유지
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    for c in SENSOR_COLS + ["machine_failure"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["ts", "machine_id"])


# 초 단위로 흔들린 타임스탬프를 분 단위로 정규화
def snap_timestamp(df: pd.DataFrame, freq: str = "min") -> pd.DataFrame:
    df = df.copy()
    df["ts"] = df["ts"].dt.round(freq)
    return df


# 동일 설비·동일 시각의 중복 데이터 중 가장 나중에 수집된 값 유지
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


# 켈빈 온도 컬럼에 혼입된 섭씨값을 탐지하고 단위 정상화
# 센서 오류값 0을 먼저 NaN으로 처리해 273.15K로 오변환되는 문제 방지
# 0.1~60℃만 섭씨 혼입값으로 판단하고, 그 외 200 미만 값은 NaN 처리
# 최종 켈빈 물리 범위는 다음 단계의 range_check()에서 검사
def detect_and_fix_temp_unit(
    df: pd.DataFrame,
    cols=("air_temp_k", "process_temp_k"),
):
    df = df.copy()
    report = {}

    for c in cols:
        if c not in df.columns:
            continue

        # 1. 센서 오류값 0을 먼저 결측으로 처리
        zero_mask = df[c].eq(0)
        df.loc[zero_mask, c] = np.nan

        # 2. 합리적인 섭씨 범위에 해당하는 값만 단위 변환
        celsius_mask = df[c].between(0.1, 60.0)
        report[c] = int(celsius_mask.sum())

        df.loc[celsius_mask, c] = df.loc[celsius_mask, c] + 273.15

        # 3. 섭씨로 볼 수 없는 200 미만 값은 결측 처리
        invalid_mask = df[c].notna() & (df[c] < 200)
        df.loc[invalid_mask, c] = np.nan

    return df, report


# 진동값은 온도와 달리 특정 값만으로 단위 오류를 확정하기 어려움
# 설비별 중앙값의 ratio배를 초과하면 단위 혼입으로 추정
# 실제 환경에서는 센서 단위 메타데이터를 우선하는 것이 적절
def detect_vibration_unit(
    df: pd.DataFrame,
    col="vibration_mms",
    factor=9.81,
    ratio=4.0,
):
    df = df.copy()

    med = df.groupby("machine_id")[col].transform("median")

    mask = df[col].notna() & (df[col] > med * ratio)
    df.loc[mask, col] = df.loc[mask, col] / factor

    return df, int(mask.sum())


# ----------------------------------------------------------------------
# 5. 물리 범위 검사
# ----------------------------------------------------------------------


# 센서별 물리적 허용 범위를 벗어난 값만 NaN 처리
# 같은 행의 다른 정상 센서값을 보존하기 위해 행 자체는 유지
def range_check(df: pd.DataFrame, rng: dict | None = None):
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


# 주변 중앙값과 MAD를 이용해 급격한 센서값 변화를 탐지
# 실제 설비 이상 신호일 수 있으므로 값을 삭제하지 않고 플래그만 기록
def hampel_flag(
    s: pd.Series,
    window: int = 11,
    n_sigma: float = 5.0,
) -> pd.Series:
    med = s.rolling(window, center=True, min_periods=3).median()

    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()

    sigma = 1.4826 * mad
    sigma = sigma.replace(0, np.nan)

    return ((s - med).abs() > n_sigma * sigma).fillna(False)


# 설비별 스파이크 여부를 컬럼으로 기록
def flag_spikes(
    df: pd.DataFrame,
    cols=None,
    window=11,
    n_sigma=5.0,
):
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


# 설비별 시간 흐름에 따라 짧은 결측 구간만 선형 보간
# 긴 결측 구간을 임의의 값으로 채우지 않도록 보간 개수 제한
def interpolate_short_gaps(
    df: pd.DataFrame,
    cols=None,
    max_gap: int = 5,
):
    cols = cols or SENSOR_COLS
    df = df.sort_values(["machine_id", "ts"]).copy()
    filled = {}

    for c in cols:
        if c not in df.columns:
            continue

        before = df[c].isna().sum()

        df[c] = df.groupby("machine_id")[c].transform(
            lambda s: s.interpolate(
                method="linear",
                limit=max_gap,
                limit_direction="both",
            )
        )

        filled[c] = int(before - df[c].isna().sum())

    return df, filled


# ----------------------------------------------------------------------
# 8. 드리프트 보정
# ----------------------------------------------------------------------


# 공정 온도와 기준 온도의 차이로 설비별 드리프트 추정
# 설비별 일일 중앙값과 전체 설비의 일일 중앙값을 비교
def estimate_drift(
    df: pd.DataFrame,
    col="process_temp_k",
    ref="air_temp_k",
):
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


# 추정된 기울기가 임계값 이상인 경우에만 센서값 보정
def correct_drift(
    df: pd.DataFrame,
    slopes: dict,
    col="process_temp_k",
    min_slope: float = 0.05,
):
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


# 설비별 빠진 측정 시각을 NaN 행으로 추가
# 실제 측정값이 없던 시점은 is_gap으로 표시
def reindex_time(
    df: pd.DataFrame,
    freq: str = "min",
) -> pd.DataFrame:
    parts = []

    for m, g in df.groupby("machine_id"):
        g = g.set_index("ts").sort_index()
        full = pd.date_range(
            g.index.min(),
            g.index.max(),
            freq=freq,
        )

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


# ----------------------------------------------------------------------
# [구현 핵심]
# ----------------------------------------------------------------------

# pandas와 NumPy를 이용해 센서 데이터의 9단계 정제 파이프라인 구현
# 단위 통일, 물리 범위 검사, 스파이크 탐지, 결측 보간, 드리프트 보정 수행
# StepLog로 단계별 행 수와 증감을 기록하도록 구현

# 온도값 200 미만을 모두 섭씨로 변환하면 오류값 0도 273.15K로 바뀌는 문제 확인
# 0을 먼저 NaN 처리하고 0.1~60℃ 범위만 섭씨 혼입값으로 판단하도록 수정
# 나머지 200 미만 값은 NaN 처리하고, 변환 후 물리 범위 검사 유지

# 추후 보완 사항
# 1. 학습·테스트 분할 이전의 전역 전처리로 인한 데이터 누수 방지
# 2. 실시간 탐지를 위한 미래 시점 참조 없는 스파이크 탐지 및 보간
# 3. 실제 센서 메타데이터를 이용한 단위 판정
