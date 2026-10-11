"""
피처 엔지니어링
===============
설비 데이터는 '지금 값'보다 '최근 흐름'이 더 많은 것을 말해줍니다.
온도 310 K가 정상인지 이상인지는, 30분 전에 305 K였는지 312 K였는지에 따라 다릅니다.

★★ 여기가 시간 누수가 가장 잘 생기는 곳입니다.
   rolling은 반드시 과거만 봐야 합니다. center=True는 미래를 봅니다 — 절대 금지.
   shift(1)까지 넣어 "현재 값도 안 보는" 엄격한 버전을 쓸지는 문제에 따라 정합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
]


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    """도메인 지식으로 만드는 파생변수. 모델보다 이게 성능을 더 올립니다."""
    df = df.copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]  # 방열 여력
    df["power_w"] = df["torque_nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60
    df["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]  # 과부하 지표(OSF)
    df["vib_per_rpm"] = df["vibration_mms"] / df["rot_speed_rpm"].replace(0, np.nan)
    return df


def add_rolling(
    df: pd.DataFrame, cols=None, windows=(10, 30, 60), shift_one: bool = True
) -> pd.DataFrame:
    """설비별 과거 window분 통계.

    shift_one=True면 현재 시점 값을 제외합니다(t-1까지만 사용).
    실시간 예측에서 '현재 값이 이미 도착했는지'가 불확실할 때 안전한 선택입니다.

    주의:
    - windows는 실제 시간 길이가 아니라 관측치 개수입니다.
    - 시간 공백이 있는 데이터에서는 30개 관측치가 30분과 다를 수 있습니다.
    - shift_one=True는 롤링 통계와 변화량 피처에 동일하게 적용합니다.
    """
    cols = cols or BASE + ["temp_diff_k", "power_w", "wear_torque"]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values(["machine_id", "ts"]).copy()
    g = df.groupby("machine_id")

    new = {}
    for c in cols:
        s = g[c]

        # 현재 시점의 센서값을 제외할지 결정합니다.
        # 변화량도 같은 기준을 사용하도록 통일합니다.
        base = s.shift(1) if shift_one else df[c]
        base_group = base.groupby(df["machine_id"])

        for w in windows:
            r = base_group.rolling(w, min_periods=max(2, w // 3))
            new[f"{c}_m{w}"] = r.mean().reset_index(level=0, drop=True)
            new[f"{c}_s{w}"] = r.std().reset_index(level=0, drop=True)

        # 변화량 (기울기 대용)
        # shift_one=True:
        #   d1  = t-1과 t-2의 차이
        #   d10 = t-1과 t-11의 차이
        #
        # shift_one=False:
        #   d1  = t와 t-1의 차이
        #   d10 = t와 t-10의 차이
        new[f"{c}_d1"] = base_group.diff()
        new[f"{c}_d10"] = base_group.diff(10)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def make_horizon_label(
    df: pd.DataFrame, horizon: int = 30, src: str = "machine_failure", out: str = "y"
) -> pd.DataFrame:
    """★ 예지보전의 진짜 목표변수: "앞으로 horizon분 안에 고장이 나는가"

    이걸 안 하고 '지금 고장인가'를 맞히면 그건 예지(prediction)가 아니라
    감지(detection)입니다. 이미 고장난 뒤에 알려주는 모델은 쓸모가 없습니다.

    기존 구현:
    - 미래 horizon개 관측치의 최댓값을 사용했습니다.
    - 관측 간격이 일정하지 않으면 실제 horizon분과 달라질 수 있었습니다.
    - 데이터 마지막 부분에서 미래 관측이 부족해도 라벨이 생성될 수 있었습니다.

    개선 구현:
    - 실제 시각을 기준으로 (현재 시각, 현재 시각 + horizon분]을 확인합니다.
    - 현재 시점의 고장 여부는 라벨에 포함하지 않습니다.
    - 미래 horizon분 전체를 관측할 수 없는 구간은 NaN으로 남깁니다.
    - 설비별로 독립적으로 계산합니다.

    주의:
    - 미래 시각 범위가 존재하더라도 그 안의 측정 누락까지 보장하지는 않습니다.
    - 데이터 수집 공백과 고장 기록 누락은 별도로 검증해야 합니다.
    """
    if horizon <= 0:
        raise ValueError("horizon은 1 이상의 분 단위 정수여야 합니다.")

    df = df.sort_values(["machine_id", "ts"]).copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    if df["ts"].isna().any():
        raise ValueError("ts에 변환할 수 없는 시각이 포함되어 있습니다.")

    if src not in df.columns:
        raise ValueError(f"고장 여부 컬럼이 없습니다: {src}")

    # 고장 여부가 알려지지 않은 값을 정상(0)으로 간주하지 않습니다.
    # 해당 구간이 미래 예측 범위에 포함되면 라벨을 확정하지 않습니다.
    failure = pd.to_numeric(df[src], errors="coerce")

    if failure.dropna().isin([0, 1]).all() is False:
        raise ValueError(f"{src}에는 0, 1 또는 결측값만 사용할 수 있습니다.")

    df[out] = np.nan
    horizon_delta = pd.Timedelta(minutes=horizon)

    for _, g in df.groupby("machine_id", sort=False):
        # 동일 설비의 시각은 정렬되어 있다고 가정합니다.
        times = g["ts"].to_numpy(dtype="datetime64[ns]")
        values = failure.loc[g.index].to_numpy(dtype=float)

        # searchsorted의 right를 사용해
        # 현재 시점은 제외하고 horizon분 끝 시점은 포함합니다.
        end_times = (g["ts"] + horizon_delta).to_numpy(dtype="datetime64[ns]")

        end_positions = np.searchsorted(
            times,
            end_times,
            side="right",
        )

        # 현재 시점 다음 행부터 검사합니다.
        starts = np.arange(len(g)) + 1

        # 고장값과 결측 개수를 누적합으로 관리해
        # 각 행마다 미래 구간을 반복 순회하지 않습니다.
        positives = np.cumsum(np.r_[0, np.nan_to_num(values, nan=0.0)])
        missing = np.cumsum(np.r_[0, np.isnan(values).astype(int)])

        positive_counts = positives[end_positions] - positives[starts]
        missing_counts = missing[end_positions] - missing[starts]

        # 각 설비에서 horizon분 이후까지 관측 범위가 존재해야 합니다.
        enough_future = end_times <= times[-1]

        # 미래 고장 여부가 결측인 구간은 결과를 확정하지 않습니다.
        valid = enough_future & (missing_counts == 0)

        labels = np.full(len(g), np.nan, dtype=float)
        labels[valid] = (positive_counts[valid] > 0).astype(float)

        df.loc[g.index, out] = labels

    return df


def build(df: pd.DataFrame, windows=(10, 30, 60), shift_one=True) -> pd.DataFrame:
    return add_rolling(add_physics(df), windows=windows, shift_one=shift_one)


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "ts",
        "machine_id",
        "type",
        "machine_failure",
        "collected_at",
        "observed_at",
        "id",
        "is_gap",
        "spike_any",
        "spike_count",
        "y",
    }

    return [
        c
        for c in df.columns
        if c not in drop
        and not c.startswith("spike_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ============================================================
# [구현 핵심]
# ============================================================

# 설비별 센서 데이터에 물리 기반 파생변수를 추가함
# 과거 관측치의 롤링 평균, 표준편차, 변화량 피처를 생성함
# 향후 30분 내 고장 여부를 목표변수로 생성함
# 모델 입력에서 식별자, 고장 여부 등 불필요한 컬럼을 제외함

# shift_one=True일 때 변화량 피처에도 동일한 과거 시점 기준을 적용함
# 미래 30개 관측치 대신 실제 미래 30분을 기준으로 고장 라벨을 계산하도록 개선함
# 미래 관측 범위가 부족하거나 고장 여부가 결측인 구간은 라벨을 NaN으로 처리함
# observed_at과 y가 모델 입력에 포함되지 않도록 명시적으로 제외함

# 추후 보완 사항
# 1. 시간 공백이 있는 데이터에서 롤링 관측치 개수와 실제 경과 시간의 차이 검증
# 2. 고장 기록 누락 및 장시간 수집 공백에 대한 라벨 유효성 검증
# 3. 학습·검증·테스트 분할 경계에서 예측 구간이 겹치지 않는지 통합 검증
