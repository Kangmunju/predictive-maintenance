"""
학습 스크립트 — 대시보드가 사용할 모델 평가 결과를 만듭니다
=========================================================
    python src/train.py

원칙 (이 프로젝트의 핵심 규칙 3가지)
  1) 목표변수는 '30분 내 고장' (감지가 아니라 예지)
  2) split은 반드시 시간순
  3) 임계값은 0.5가 아니라 비용 최소점

추가 검증
  - 양성률 기준선, 단순 이력 규칙, RandomForest의 AP 비교
  - 단순 규칙은 현재 고장 여부가 아닌 과거 고장 이력만 사용

현재 남아 있는 보완 과제
  - 학습·테스트 경계의 시간 누수 방지
  - 전처리 통계의 학습 데이터 기준 적용
  - 테스트 데이터와 분리된 임계값 선정
  - 고장 에피소드 단위의 경보 및 비용 평가
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
    2. shift(1)로 현재 시점의 고장 여부를 제외합니다.
    3. 이전 30개 관측치의 고장 비율을 계산합니다.
    4. 계산 결과를 원래 데이터의 행 순서에 맞춰 반환합니다.

    주의:
    - 이전 30개 관측치가 반드시 실제 30분을 의미하지는 않습니다.
    - 고장 이력이 예측 시점에 확인 가능하다는 가정이 필요합니다.
    - 이 점수는 확률 보정된 예측 확률이 아닙니다.
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

    clean, log, rep = run_pipeline(raw, verbose=True)

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

    # [추가]
    # 설비별 이전 30개 관측치의 고장 이력을 이용해
    # 단순 위험 점수를 계산합니다.
    #
    # RandomForest와 동일한 테스트 데이터에서
    # AP를 비교하기 위한 기준 모델입니다.

    naive_scores = make_naive_score(d)

    # --------------------------------------------------
    # 5. 학습 및 테스트 데이터 분할
    # --------------------------------------------------

    X = d[feat].values
    y = d["y"].values.astype(int)

    # 기존 시간순 75:25 행 기준 분할
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

    # [추가]
    # 세 지표 모두 동일한 테스트 데이터를 기준으로 계산합니다.

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
    # 8. 비용 최소 임계값 계산
    # --------------------------------------------------

    # 기존 방식 유지
    #
    # 현재는 1분 단위 행별로 비용을 계산합니다.
    # 고장 사건을 여러 번 계산할 가능성이 있으므로
    # 후속 에피소드 단위 평가에서 수정할 예정입니다.
    #
    # 또한 테스트 데이터에서 임계값을 선택하는 문제는
    # 후속 평가 방식 보강에서 수정할 예정입니다.

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
    # 9. 평가 지표 정리
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
        # [추가] 단순 규칙 비교 지표
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
        "cost_at_threshold": int(costs[i]),
        "cost_at_0.5": int(costs[int(np.argmin(np.abs(ths - 0.5)))]),
    }

    print("\n[성능]")

    for k, v in metrics.items():
        print(f"  {k:<28} {v}")

    # --------------------------------------------------
    # 10. 변수 중요도 확인
    # --------------------------------------------------

    imp = pd.Series(
        mdl.feature_importances_,
        index=feat,
    ).sort_values(ascending=False)

    print("\n[변수 중요도 상위 10]")
    print(imp.head(10).round(4).to_string())

    # --------------------------------------------------
    # 11. 결과 파일 저장
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
    pd.DataFrame(
        {
            "ts": d["ts"].iloc[cut:].values,
            "machine_id": d["machine_id"].iloc[cut:].values,
            "y": yt,
            "prob": p,
        }
    ).to_csv(
        outdir / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 12. 실행 결과 안내
    # --------------------------------------------------

    print("\n저장: models/metrics.json, feature_importance.csv, test_predictions.csv")

    # [수정]
    # 모델 파일을 저장하지 않는 현재 구조를 설명합니다.
    # scikit-learn 버전 차이만을 이유로 단정하지 않습니다.

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


# 원본 183,692행 (43개 파일)
#            단계      행수      증감
#      0. 원본 수신  183692       0
#      1. 타입 강제  183692       0
#   2. 타임스탬프 스냅  183692       0
#      3. 중복 제거  177402   -6290
#  4a. 온도 단위 통일  177402       0
#  4b. 진동 단위 통일  177402       0
# 5. 물리범위 → NaN  177402       0
#   6. 스파이크 플래그  177402       0
#   7. 짧은 결측 보간  177402       0
#    8. 드리프트 보정  177402       0
#    9. 시간축 재색인 4371986 4194584

# train 133,004 / test 44,335 | train 양성률 22.62% / test 양성률 21.57%

# [AP 비교: 동일 테스트 데이터]
#   양성률 기준선          0.2157
#   단순 이력 규칙         0.3707
#   RandomForest          0.7649
#   RF - 단순 규칙         +0.3942

# [성능]
#   n_rows                       177339
#   n_features                   100
#   horizon_min                  30
#   train_end                    2026-09-27 07:41:00
#   test_start                   2026-09-27 07:41:00
#   positive_rate_test           0.2157
#   roc_auc                      0.9052
#   pr_auc                       0.7649
#   pr_auc_naive                 0.3707
#   ap_improvement_over_naive    0.3942
#   naive_method                 previous_30_observations_failure_rate
#   threshold                    0.045
#   precision                    0.3526
#   recall                       0.9557
#   f1                           0.5151
#   cost_at_threshold            8425100000
#   cost_at_0.5                  28831800000

# [변수 중요도 상위 10]
# rot_speed_rpm_m10    0.0366
# wear_torque          0.0353
# torque_nm_m10        0.0328
# rot_speed_rpm_m30    0.0319
# vib_per_rpm          0.0293
# torque_nm_m30        0.0271
# wear_torque_m10      0.0256
# power_w_m10          0.0253
# rot_speed_rpm_m60    0.0227
# current_a_m10        0.0221

# 저장: models/metrics.json, feature_importance.csv, test_predictions.csv
# ★ 현재 파이프라인은 학습된 모델 파일을 저장하지 않고, 평가 지표와 예측 결과를 저장합니다.
#   대시보드는 저장된 예측 결과 CSV를 사용하며, 모델 재학습은 이 스크립트로 수행합니다.
#   모델 파일 저장이 필요한 경우에는 scikit-learn 버전 관리와 직렬화 방식을 별도로 고려해야 합니다.


# ============================================================

# RandomForestClassifier로 향후 30분 내 고장 여부를 예측함
# predict_proba()로 고장 위험 확률을 계산함

# 센서 및 롤링 피처 100개를 사용
# 시간순 75:25 분할 후 ROC-AUC, AP, Precision, Recall, F1으로 성능을 평가함

# FN/FP 비용을 다르게 설정해 총비용이 최소인 예측 임계값을 탐색하도록 구현함

# RandomForest 성능만으로는 단순 규칙 대비 성능 향상을 판단하기 어려움을 느낌
# 설비별 과거 30개 관측치의 고장 비율을 기준 모델로 추가하였음
# groupby() + shift(1) + rolling().mean()으로 작성해 구현
# 동일 테스트 구간에서 AP를 비교하도록 개선하였음

# .pkl 미저장 이유를 sklearn 버전 문제로 단정해버렸음
# 문제 해결을 위해 평가 지표와 예측 결과를 저장하는 현재 구조로 설명을 변경함

# 추후 보완 사항
# 1. 시간 분할 경계 및 전처리 누수 방지
# 2. 고장 에피소드 단위 비용 평가
# 3. 검증 데이터 기반 임계값 선정
