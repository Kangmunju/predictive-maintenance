"""
학습 스크립트 — 대시보드가 사용할 모델 평가 결과를 만듭니다
=========================================================

    python3 -m src.train

원칙
  1) 목표변수는 '향후 30분 내 고장' (감지가 아니라 예지)
  2) 학습·테스트 데이터는 시간순으로 분할
  3) FN/FP 비용을 고려해 예측 임계값 비교

추가 검증
  - 양성률 기준선, 단순 이력 규칙, RandomForest AP 비교
  - 실제 고장 에피소드와 독립 알람 기준 운영 성능 평가

현재 남아 있는 보완 과제
  - 학습·테스트 경계의 시간 누수 방지
  - 전처리 통계의 학습 데이터 기준 적용
  - 테스트 데이터와 분리된 임계값 선정
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402
from clean import run_pipeline  # noqa: E402
from episode_eval import episode_metrics  # noqa: E402


HORIZON = 30
COST_FN = 8_000_000
COST_FP = 300_000
SEED = 42


def load_history() -> pd.DataFrame:
    """data/history의 CSV 파일을 모두 읽어 하나의 데이터프레임으로 합칩니다."""

    files = sorted((ROOT / "data" / "history").glob("*.csv"))

    if not files:
        raise SystemExit(
            "data/history/*.csv 가 없습니다. collector.py를 먼저 돌리세요."
        )

    return pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True,
    )


def make_naive_score(d: pd.DataFrame) -> pd.Series:
    """
    단순 이력 기반 기준 모델의 위험 점수를 계산합니다.

    1. 설비별로 데이터를 시간순 정렬합니다.
    2. shift(1)로 현재 고장 여부를 제외합니다.
    3. 이전 30개 관측치의 고장 비율을 계산합니다.
    4. 원래 데이터의 행 순서에 맞춰 반환합니다.

    주의:
    - 이전 30개 관측치가 반드시 실제 30분을 뜻하지는 않습니다.
    - 과거 고장 이력이 예측 시점에 확인 가능하다고 가정합니다.
    """

    ordered = d.sort_values(
        ["machine_id", "ts"],
        kind="stable",
    ).copy()

    history = (
        ordered.groupby("machine_id")["machine_failure"]
        .transform(lambda s: s.fillna(0).shift(1).rolling(30, min_periods=1).mean())
        .fillna(0.0)
    )

    return history.reindex(d.index).astype(float)


def main() -> int:
    # --------------------------------------------------
    # 1. 데이터 불러오기
    # --------------------------------------------------

    raw = load_history()

    n_files = len(list((ROOT / "data" / "history").glob("*.csv")))

    print(f"원본 {len(raw):,}행 ({n_files}개 파일)")

    # --------------------------------------------------
    # 2. 데이터 정제
    # --------------------------------------------------

    clean, log, rep = run_pipeline(
        raw,
        verbose=True,
    )

    clean = clean[~clean["is_gap"].astype(bool)].copy()

    # --------------------------------------------------
    # 3. 목표변수 및 파생변수 생성
    # --------------------------------------------------

    d = F.make_horizon_label(
        clean,
        horizon=HORIZON,
    )

    d = F.build(
        d,
        windows=(10, 30, 60),
        shift_one=True,
    )

    feat = [c for c in F.feature_columns(d) if c not in ("y", "machine_failure")]

    d = d.dropna(subset=feat + ["y"]).sort_values("ts").reset_index(drop=True)

    if len(d) < 500 or d["y"].nunique() < 2:
        raise SystemExit(
            f"학습에 쓸 데이터가 부족합니다({len(d)}행). 수집을 더 하세요."
        )

    # --------------------------------------------------
    # 4. 단순 이력 기반 기준 모델
    # --------------------------------------------------

    # 현재 시점의 고장 여부를 사용하지 않고
    # 설비별 이전 30개 관측치로 위험 점수를 계산합니다.

    naive_scores = make_naive_score(d)

    # --------------------------------------------------
    # 5. 학습 및 테스트 데이터 분할
    # --------------------------------------------------

    X = d[feat].values
    y = d["y"].values.astype(int)

    # 기존 시간순 75:25 행 기준 분할 유지
    # 같은 시각의 설비 데이터가 양쪽에 나뉠 수 있으므로
    # 후속 시간 누수 보강 단계에서 수정할 예정입니다.

    cut = int(len(d) * 0.75)

    print(
        f"\ntrain {cut:,} / test {len(d) - cut:,} | "
        f"train 양성률 {y[:cut].mean() * 100:.2f}% / "
        f"test 양성률 {y[cut:].mean() * 100:.2f}%"
    )

    # --------------------------------------------------
    # 6. RandomForest 학습 및 예측
    # --------------------------------------------------

    mdl = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    mdl.fit(
        X[:cut],
        y[:cut],
    )

    p = mdl.predict_proba(X[cut:])[:, 1]
    yt = y[cut:]

    # --------------------------------------------------
    # 7. 기준 모델과 RandomForest AP 비교
    # --------------------------------------------------

    # 모든 AP를 동일한 테스트 데이터에서 계산합니다.

    naive_test = naive_scores.iloc[cut:].to_numpy()

    ap_baseline = float(yt.mean())

    ap_naive = float(
        average_precision_score(
            yt,
            naive_test,
        )
    )

    ap_rf = float(
        average_precision_score(
            yt,
            p,
        )
    )

    ap_improvement = ap_rf - ap_naive

    print("\n[AP 비교: 동일 테스트 데이터]")
    print(f"  양성률 기준선          {ap_baseline:.4f}")
    print(f"  단순 이력 규칙         {ap_naive:.4f}")
    print(f"  RandomForest          {ap_rf:.4f}")
    print(f"  RF - 단순 규칙         {ap_improvement:+.4f}")

    # --------------------------------------------------
    # 8. 기존 행 단위 비용 최소 임계값 계산
    # --------------------------------------------------

    # 기존 결과와 비교하기 위해 행 단위 비용을 유지합니다.
    # 동일한 고장 사건을 여러 행에서 계산할 수 있으므로
    # 실제 운영 비용으로 해석하지 않습니다.
    #
    # 테스트 데이터로 임계값을 선택하는 한계는
    # 후속 검증 데이터 분리 단계에서 해결할 예정입니다.

    ths = np.linspace(0.01, 0.99, 197)
    costs = []

    for t in ths:
        tn, fp, fn, tp = confusion_matrix(
            yt,
            (p >= t).astype(int),
            labels=[0, 1],
        ).ravel()

        costs.append(fn * COST_FN + fp * COST_FP)

    i = int(np.argmin(costs))
    th = float(ths[i])

    # --------------------------------------------------
    # 9. 테스트 예측 결과 구성
    # --------------------------------------------------

    # [추가]
    # y: 향후 30분 내 고장 여부
    # machine_failure: 현재 시점의 실제 고장 여부
    #
    # 에피소드 평가는 실제 고장 발생 시각을
    # 확인해야 하므로 machine_failure가 필요합니다.

    test_predictions = pd.DataFrame(
        {
            "ts": d["ts"].iloc[cut:].values,
            "machine_id": d["machine_id"].iloc[cut:].values,
            "y": yt,
            "machine_failure": (d["machine_failure"].iloc[cut:].values),
            "prob": p,
        }
    )

    # --------------------------------------------------
    # 10. 고장 에피소드 단위 평가
    # --------------------------------------------------

    # [추가]
    # 실제 고장 발생 시각을 기준으로 사건을 구분하고
    # 10분 이내 반복 경고를 하나의 알람으로 묶습니다.
    #
    # 기존 행 단위 임계값을 그대로 적용하여
    # 두 평가 방식의 결과를 비교합니다.

    episode_result = episode_metrics(
        test_predictions,
        threshold=th,
        horizon_min=HORIZON,
        alarm_gap_min=10,
        cost_fn=COST_FN,
        cost_fp=COST_FP,
    )

    episode_result_05 = episode_metrics(
        test_predictions,
        threshold=0.5,
        horizon_min=HORIZON,
        alarm_gap_min=10,
        cost_fn=COST_FN,
        cost_fp=COST_FP,
    )

    print("\n[고장 에피소드 단위 평가]")
    print(f"  평가 임계값             {th:.3f}")
    print(f"  고장 에피소드           {episode_result['failure_episodes']}")
    print(f"  탐지 성공               {episode_result['detected_episodes']}")
    print(f"  미탐지 고장             {episode_result['missed_episodes']}")
    print(f"  에피소드 탐지율         {episode_result['episode_recall']}")
    print(f"  독립 알람               {episode_result['alarm_groups']}")
    print(f"  오경보                  {episode_result['false_alarms']}")
    print(f"  일평균 오경보           {episode_result['false_alarms_per_day']}")
    print(f"  에피소드 비용           {episode_result['episode_cost']:,}원")
    print(f"  임계값 0.5 비용         {episode_result_05['episode_cost']:,}원")

    # --------------------------------------------------
    # 11. 평가 지표 정리
    # --------------------------------------------------

    metrics = {
        "n_rows": int(len(d)),
        "n_features": len(feat),
        "horizon_min": HORIZON,
        "train_end": str(d["ts"].iloc[cut - 1]),
        "test_start": str(d["ts"].iloc[cut]),
        "positive_rate_test": round(ap_baseline, 4),
        "roc_auc": round(
            float(roc_auc_score(yt, p)),
            4,
        ),
        "pr_auc": round(ap_rf, 4),
        "pr_auc_naive": round(ap_naive, 4),
        "ap_improvement_over_naive": round(
            ap_improvement,
            4,
        ),
        "naive_method": ("previous_30_observations_failure_rate"),
        "threshold": round(th, 3),
        "precision": round(
            float(
                precision_score(
                    yt,
                    (p >= th).astype(int),
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    yt,
                    (p >= th).astype(int),
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    yt,
                    (p >= th).astype(int),
                    zero_division=0,
                )
            ),
            4,
        ),
        # 기존 행 단위 비용
        "cost_at_threshold": int(costs[i]),
        "cost_at_0.5": int(costs[int(np.argmin(np.abs(ths - 0.5)))]),
        # [추가] 에피소드 단위 평가 결과
        "episode_evaluation": episode_result,
        "episode_evaluation_at_0.5": episode_result_05,
    }

    print("\n[성능]")

    for k, v in metrics.items():
        if k.startswith("episode_evaluation"):
            continue

        print(f"  {k:<28} {v}")

    # --------------------------------------------------
    # 12. 변수 중요도 확인
    # --------------------------------------------------

    imp = pd.Series(
        mdl.feature_importances_,
        index=feat,
    ).sort_values(ascending=False)

    print("\n[변수 중요도 상위 10]")
    print(imp.head(10).round(4).to_string())

    # --------------------------------------------------
    # 13. 결과 파일 저장
    # --------------------------------------------------

    outdir = ROOT / "models"
    outdir.mkdir(exist_ok=True)

    # 모델 평가 지표
    (outdir / "metrics.json").write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 변수 중요도 상위 30개
    imp.head(30).to_csv(
        outdir / "feature_importance.csv",
        header=["importance"],
    )

    # 테스트 구간의 예측 결과
    test_predictions.to_csv(
        outdir / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 14. 실행 결과 안내
    # --------------------------------------------------

    print("\n저장: models/metrics.json, feature_importance.csv, test_predictions.csv")

    print(
        "★ 현재 파이프라인은 학습된 모델 파일을 저장하지 않고, "
        "평가 지표와 예측 결과를 저장합니다."
    )

    print(
        "  대시보드는 저장된 예측 결과 CSV를 사용하며, "
        "모델 재학습은 이 스크립트로 수행합니다."
    )

    print(
        "  모델 파일 저장이 필요한 경우에는 "
        "scikit-learn 버전 관리와 직렬화 방식을 별도로 고려해야 합니다."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ============================================================
# [구현 핵심]
# ============================================================

# RandomForestClassifier로 향후 30분 내 고장 여부를 예측함
# predict_proba()로 고장 위험 확률을 계산함
# 센서 및 롤링 피처를 사용해 예측 모델을 학습함

# 설비별 과거 30개 관측치의 고장 비율을 단순 기준 모델로 구현함
# groupby() + shift(1) + rolling().mean()으로 현재 고장 정보 사용을 방지함
# 동일 테스트 구간에서 단순 규칙과 RandomForest의 AP를 비교함

# 기존 비용 평가는 1분 단위 행마다 FN/FP 비용을 적용해 같은 고장을 중복 계산할 수 있음
# 이를 보완하기 위해 고장 에피소드 단위 평가를 추가함
# y는 향후 30분 내 고장 여부, machine_failure는 현재 시점의 실제 고장 여부를 의미함
# 실제 고장 시각을 식별하기 위해 test_predictions에 machine_failure 컬럼을 추가함
# 직전 고장과 30분 이내로 발생한 반복 고장을 하나의 에피소드로 묶음
# 동일 설비에서 10분 이내 반복된 경고를 하나의 독립 알람으로 묶음
# 미탐지 고장 에피소드 수와 오경보 수를 기준으로 운영 비용을 계산함

# episode_metrics()를 두 번 호출해 임계값 th와 기본 임계값 0.5를 각각 평가함
# th는 기존 행 단위 비용을 최소화한 임계값이며, 에피소드 비용의 최적 임계값은 아님
# 기존 행 단위 비용은 cost_at_threshold와 cost_at_0.5에 유지함
# 에피소드 평가 결과는 episode_evaluation과 episode_evaluation_at_0.5에 별도로 저장함
# 두 평가 방식의 결과를 metrics.json에서 구분해 확인할 수 있도록 구현함

# 추후 보완 사항
# 1. 시간 분할 경계 및 전처리 누수 방지
# 2. 검증 데이터 기반 임계값 선정
# 3. 테스트 경계의 고장 이력 부족 문제 보완
# 4. 고장 에피소드와 독립 알람의 연결 기준 검증
