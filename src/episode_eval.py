"""
고장 에피소드 단위 평가
======================

1. 실제 고장 시각을 기준으로 독립 고장 에피소드를 구분
2. 동일 설비의 반복 경고를 하나의 알람으로 묶기
3. 알람 활성 구간과 고장 전 30분 구간의 겹침 여부 확인
4. 알람과 고장 에피소드를 일대일로 연결
5. 미탐지 고장과 오경보를 기준으로 운영 비용 계산

평가 기준
- 직전 고장 기록과 30분 이내로 이어지는 고장은 하나의 에피소드
- 동일 설비에서 직전 경고와 10분 이내로 이어지는 경고는 하나의 알람
- 알람 활성 구간이 고장 전 30분 구간과 겹치면 탐지 후보
- 하나의 알람은 최대 하나의 고장 에피소드에 연결
- 테스트 시작 후 첫 30분의 고장은 평가에서 제외
- 테스트 종료 전 마지막 30분의 알람은 오경보 판정에서 제외

주의
- y는 향후 30분 내 고장 여부를 나타내는 학습용 라벨
- 에피소드 식별에는 실제 고장 여부인 machine_failure를 사용
- 연속 경고 그룹의 시작~종료 시각을 활성 구간으로 간주
"""

from __future__ import annotations

import pandas as pd


HORIZON_MIN = 30
ALARM_GAP_MIN = 10

COST_FN = 8_000_000
COST_FP = 300_000


def prepare_data(df: pd.DataFrame, required: set) -> pd.DataFrame:
    """필수 컬럼을 확인하고 설비별 시간순으로 정렬합니다."""

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])

    if d["ts"].isna().any():
        raise ValueError("ts에 유효하지 않은 시각이 있습니다.")

    return d.sort_values(
        ["machine_id", "ts"],
        kind="stable",
    ).reset_index(drop=True)


def find_failure_episodes(
    df: pd.DataFrame,
    horizon_min: int = HORIZON_MIN,
) -> pd.DataFrame:
    """
    실제 고장 기록을 독립 고장 에피소드로 구분합니다.

    직전 고장 기록과의 간격이 horizon_min을 초과하면
    새로운 에피소드로 판단합니다.
    """

    d = prepare_data(
        df,
        {"ts", "machine_id", "machine_failure"},
    )

    failures = (
        d.loc[
            d["machine_failure"].eq(1),
            ["machine_id", "ts"],
        ]
        .drop_duplicates()
        .copy()
    )

    if failures.empty:
        return pd.DataFrame(columns=["machine_id", "failure_time"])

    prev_failure = failures.groupby("machine_id")["ts"].shift(1)

    new_episode = prev_failure.isna() | (
        failures["ts"] - prev_failure > pd.Timedelta(minutes=horizon_min)
    )

    episodes = failures.loc[new_episode].copy()

    return episodes.rename(columns={"ts": "failure_time"}).reset_index(drop=True)


def group_alarms(
    df: pd.DataFrame,
    threshold: float,
    gap_min: int = ALARM_GAP_MIN,
) -> pd.DataFrame:
    """
    동일 설비의 반복 경고를 하나의 알람으로 묶습니다.

    직전 경고와의 간격이 gap_min을 초과하면
    새로운 알람으로 판단합니다.

    alarm_start: 알람 그룹의 첫 경고 시각
    alarm_end: 알람 그룹의 마지막 경고 시각
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold는 0~1 사이여야 합니다.")

    d = prepare_data(
        df,
        {"ts", "machine_id", "prob"},
    )

    alarms = (
        d.loc[
            d["prob"].ge(threshold),
            ["machine_id", "ts"],
        ]
        .drop_duplicates()
        .copy()
    )

    if alarms.empty:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "alarm_start",
                "alarm_end",
            ]
        )

    prev_alarm = alarms.groupby("machine_id")["ts"].shift(1)

    new_alarm = prev_alarm.isna() | (
        alarms["ts"] - prev_alarm > pd.Timedelta(minutes=gap_min)
    )

    alarms["alarm_group"] = new_alarm.groupby(alarms["machine_id"]).cumsum()

    grouped = alarms.groupby(
        ["machine_id", "alarm_group"],
        as_index=False,
    ).agg(
        alarm_start=("ts", "min"),
        alarm_end=("ts", "max"),
    )

    return grouped[["machine_id", "alarm_start", "alarm_end"]].reset_index(drop=True)


def episode_metrics(
    df: pd.DataFrame,
    threshold: float,
    horizon_min: int = HORIZON_MIN,
    alarm_gap_min: int = ALARM_GAP_MIN,
    cost_fn: int = COST_FN,
    cost_fp: int = COST_FP,
) -> dict:
    """
    고장 에피소드와 알람 활성 구간을 비교하여 평가합니다.

    탐지 성공:
    - 동일 설비에서 발생한 알람
    - 알람 활성 구간과 고장 전 예측 구간이 겹침
    - 다른 고장 에피소드에 이미 연결되지 않은 알람

    오경보:
    - 평가 가능한 구간에서 시작된 알람
    - 평가 대상 고장과 연결되지 않은 알람

    경계 처리:
    - 테스트 시작 후 첫 horizon_min 동안의 고장은 제외
    - 테스트 종료 전 마지막 horizon_min 동안의 알람은
      이후 고장 발생 여부를 알 수 없어 오경보에서 제외
    """

    d = prepare_data(
        df,
        {
            "ts",
            "machine_id",
            "machine_failure",
            "prob",
        },
    )

    if d.empty:
        raise ValueError("평가할 데이터가 없습니다.")

    if horizon_min <= 0 or alarm_gap_min <= 0:
        raise ValueError("시간 간격은 0보다 커야 합니다.")

    episodes = find_failure_episodes(
        d,
        horizon_min=horizon_min,
    )

    alarms = group_alarms(
        d,
        threshold=threshold,
        gap_min=alarm_gap_min,
    )

    horizon = pd.Timedelta(minutes=horizon_min)

    # --------------------------------------------------
    # 1. 설비별 평가 시작·종료 시각 확인
    # --------------------------------------------------

    boundaries = d.groupby("machine_id")["ts"].agg(test_start="min", test_end="max")

    episodes = episodes.join(
        boundaries,
        on="machine_id",
    )

    episodes["eligible"] = episodes["failure_time"] >= episodes["test_start"] + horizon

    alarms = alarms.copy()
    alarms["alarm_id"] = range(len(alarms))

    alarms = alarms.join(
        boundaries,
        on="machine_id",
    )

    alarms["fp_eligible"] = alarms["alarm_start"] <= alarms["test_end"] - horizon

    # --------------------------------------------------
    # 2. 고장 에피소드와 알람 연결
    # --------------------------------------------------

    used_alarm_ids = set()
    excluded_alarm_ids = set()

    detected = 0

    episodes = episodes.sort_values(
        ["failure_time", "machine_id"],
        kind="stable",
    )

    for _, episode in episodes.iterrows():
        failure_time = episode["failure_time"]
        window_start = failure_time - horizon

        # [수정 핵심]
        #
        # 기존:
        # alarm_start >= window_start
        # alarm_start < failure_time
        #
        # 수정:
        # alarm_start < failure_time
        # alarm_end >= window_start
        #
        # 알람이 예측 구간 이전에 시작되었더라도
        # 고장 전 30분 구간까지 유지되었다면
        # 탐지 후보로 인정합니다.

        candidates = alarms.loc[
            (alarms["machine_id"] == episode["machine_id"])
            & (alarms["alarm_start"] < failure_time)
            & (alarms["alarm_end"] >= window_start)
        ].sort_values(["alarm_end", "alarm_start"])

        if not bool(episode["eligible"]):
            # 테스트 시작 경계에 있는 고장과 연결 가능한
            # 알람은 오경보 판정에서도 제외합니다.
            excluded_alarm_ids.update(candidates["alarm_id"].tolist())
            continue

        # 이미 다른 고장에 연결된 알람은 재사용하지 않음
        available = candidates.loc[
            ~candidates["alarm_id"].isin(used_alarm_ids)
            & ~candidates["alarm_id"].isin(excluded_alarm_ids)
        ]

        if available.empty:
            continue

        # 고장 이전에 가장 최근까지 경고가 이어진
        # 알람 그룹을 우선 선택합니다.
        selected = available.iloc[-1]

        used_alarm_ids.add(int(selected["alarm_id"]))

        detected += 1

    # --------------------------------------------------
    # 3. 미탐지 고장 계산
    # --------------------------------------------------

    n_episodes = int(episodes["eligible"].sum())
    missed = n_episodes - detected

    # --------------------------------------------------
    # 4. 독립 오경보 계산
    # --------------------------------------------------

    false_alarm_rows = alarms.loc[
        alarms["fp_eligible"]
        & ~alarms["alarm_id"].isin(used_alarm_ids)
        & ~alarms["alarm_id"].isin(excluded_alarm_ids)
    ]

    false_alarms = len(false_alarm_rows)

    # --------------------------------------------------
    # 5. 일평균 오경보 및 운영 비용 계산
    # --------------------------------------------------

    duration_days = (d["ts"].max() - d["ts"].min()).total_seconds() / 86400.0

    duration_days = max(
        duration_days,
        1 / 1440,
    )

    total_cost = missed * cost_fn + false_alarms * cost_fp

    return {
        "threshold": round(float(threshold), 4),
        "failure_episodes": int(n_episodes),
        "detected_episodes": int(detected),
        "missed_episodes": int(missed),
        "episode_recall": (round(detected / n_episodes, 4) if n_episodes else None),
        "alarm_groups": int(len(alarms)),
        "false_alarms": int(false_alarms),
        "false_alarms_per_day": round(
            false_alarms / duration_days,
            4,
        ),
        "episode_cost": int(total_cost),
    }


# ============================================================
# [에피소드 평가 결과 - 알람 활성 구간 기준]
# ============================================================

# 고장 에피소드: 190건
# 탐지 성공: 150건
# 미탐지: 40건
# 에피소드 탐지율: 78.95%
# 독립 알람: 490건
# 오경보: 339건
# 일평균 오경보: 28.9434건
# 에피소드 비용: 421,700,000원
# 임계값 0.5 기준 비용: 510,600,000원
# RandomForest AP: 0.7664
# 단순 이력 기준 AP: 0.3707
# 평가 임계값: 0.03


# ============================================================
# [구현 핵심]
# ============================================================

# groupby()와 shift()로 설비별 직전 고장 시각을 확인함
# 직전 고장과 30분 이내로 이어지는 고장을 하나의 에피소드로 구분함
# 예측 확률이 임계값 이상인 시각을 경고로 판단함
# 동일 설비에서 직전 경고와 10분 이내로 이어지는 경고를 하나의 알람으로 묶음

# 기존 에피소드 평가에서는 하나의 알람이 여러 고장 에피소드에 중복 연결될 수 있었음
# 알람에 고유 번호를 부여하고 사용 여부를 관리해 일대일 연결로 변경함

# 알람 시작 시각만 기준으로 평가하면 장시간 지속된 경고를 미탐지로 처리할 수 있음
# alarm_start와 alarm_end를 사용해 알람 활성 구간을 확인하도록 변경함
# 알람 활성 구간이 고장 전 30분 예측 구간과 겹치면 탐지 후보로 인정함

# 테스트 시작 후 첫 30분 동안 발생한 고장은 과거 경고 이력이 부족해 평가에서 제외함
# 테스트 종료 전 마지막 30분 동안 시작된 알람은 미래 고장 여부를 알 수 없어 오경보 판정에서 제외함
# 미탐지 고장 에피소드 수와 독립 오경보 수에 각각 비용을 적용함

# 알람 시작 시각만 사용하는 방식에서는 탐지율 20.00%, 오경보 451건을 기록함
# 알람 활성 구간을 반영한 평가에서는 고장 에피소드 190건 중 150건을 탐지함
# 에피소드 탐지율 78.95%, 독립 알람 490건, 오경보 339건을 기록함
# 에피소드 비용은 421,700,000원으로 계산됨
# 시작 시각 기준 평가보다 탐지율이 개선되었으며 일대일 연결 규칙도 유지함
# 모델의 AP는 0.7664로 동일해 평가 정책의 변화가 운영 지표에 미치는 영향을 확인함

# 추후 보완 사항
# 1. 장시간 지속 알람의 재발령 및 해제 정책 정의
# 2. 실제 고장 전체 이력과 예측 가능 시점을 분리한 평가
# 3. 데이터 누락 구간 및 테스트 경계를 고려한 평가 기간 보정
# 4. 검증 데이터에서 임계값을 선택하고 테스트 데이터로 최종 평가
