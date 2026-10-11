"""
학습 스크립트 — 대시보드가 사용할 모델 평가 결과를 만듭니다
=========================================================

    python3 -m src.train

원칙
  1) 목표변수는 '향후 30분 내 고장' (감지가 아니라 예지)
  2) 학습·검증·테스트 데이터는 시간순으로 분할
  3) FN/FP 비용을 고려해 예측 임계값 비교

추가 검증
  - 양성률 기준선, 단순 이력 규칙, RandomForest AP 비교
  - 실제 고장 에피소드와 독립 알람 기준 운영 성능 평가

데이터 누수 방지
  - 미래 센서값을 참조하지 않는 causal 정제 사용
  - 동일 시각의 설비 데이터가 서로 다른 구간에 나뉘지 않도록 분할
  - 학습·검증 경계에서 향후 30분의 정답 정보가 넘어가지 않도록 제외
  - 검증 데이터로 임계값을 선정하고 테스트 데이터에서는 고정
  - 테스트 데이터는 최종 성능 평가에만 사용

현재 남아 있는 보완 과제
  - 전처리 중 학습 데이터로 추정한 통계의 고정 적용 방식 검토
  - 테스트 경계의 고장 이력 부족 문제 보완
  - 고장 에피소드와 독립 알람의 연결 기준 검증
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

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20


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


def split_by_time(
    d: pd.DataFrame,
    horizon_min: int = HORIZON,
):
    """
    시간 기준으로 학습·검증·테스트 데이터를 분리합니다.

    1. 고유 타임스탬프를 기준으로 60:20:20 분할합니다.
    2. 동일 시각의 설비 데이터는 같은 구간에 배치합니다.
    3. 학습·검증 구간 끝에서 horizon_min만큼의 행을 제외합니다.

    예:
    - 검증 시작이 12:00이라면 학습 데이터의 마지막 시점은
      11:30 이전이어야 합니다.
    - 11:40의 정답은 12:10까지의 고장 정보를 사용하므로
      학습 데이터에 포함하지 않습니다.

    주의:
    - 경계에서 제외한 행은 학습이나 임계값 선정에 사용하지 않습니다.
    - 테스트 구간은 미래 정답이 확보된 행만 평가합니다.
    """

    d = d.sort_values(
        ["ts", "machine_id"],
        kind="stable",
    ).reset_index(drop=True)

    times = pd.Index(d["ts"].drop_duplicates().sort_values())

    if len(times) < 3:
        raise SystemExit("시간순 분할에 필요한 서로 다른 타임스탬프가 부족합니다.")

    train_idx = int(len(times) * TRAIN_RATIO)
    valid_idx = int(len(times) * (TRAIN_RATIO + VALID_RATIO))

    if not (0 < train_idx < valid_idx < len(times)):
        raise SystemExit("학습·검증·테스트 분할 경계를 생성할 수 없습니다.")

    valid_start = times[train_idx]
    test_start = times[valid_idx]

    embargo = pd.Timedelta(minutes=horizon_min)

    # 향후 30분 정답이 검증 구간으로 넘어가는 학습 행 제외
    train = d.loc[d["ts"] < valid_start - embargo].copy()

    # 향후 30분 정답이 테스트 구간으로 넘어가는 검증 행 제외
    valid = d.loc[(d["ts"] >= valid_start) & (d["ts"] < test_start - embargo)].copy()

    test = d.loc[d["ts"] >= test_start].copy()

    for name, part in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:
        if part.empty:
            raise SystemExit(
                f"{name} 데이터가 비어 있습니다. 수집 기간 또는 분할 비율을 확인하세요."
            )

    return train, valid, test


def calculate_row_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> int:
    """
    기존 행 단위 FN/FP 비용을 계산합니다.

    동일한 고장 사건을 여러 행에서 계산할 수 있으므로
    실제 운영 비용으로 해석하지 않습니다.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        (probabilities >= threshold).astype(int),
        labels=[0, 1],
    ).ravel()

    return int(fn * COST_FN + fp * COST_FP)


def choose_threshold(
    y_valid: np.ndarray,
    p_valid: np.ndarray,
):
    """
    검증 데이터에서만 비용 최소 임계값을 선택합니다.

    테스트 데이터의 정답과 예측 확률은
    임계값 선택에 사용하지 않습니다.
    """

    ths = np.linspace(0.01, 0.99, 197)

    costs = [calculate_row_cost(y_valid, p_valid, float(t)) for t in ths]

    i = int(np.argmin(costs))
    th = float(ths[i])

    return th, int(costs[i])


def safe_roc_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    """
    평가 구간에 양성 또는 음성 한 종류만 있는 경우
    ROC-AUC를 계산할 수 없으므로 None을 반환합니다.
    """

    if len(np.unique(y_true)) < 2:
        return None

    return round(
        float(roc_auc_score(y_true, probabilities)),
        4,
    )


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

    # [수정]
    # 기존 run_pipeline()은 전체 기간 중앙값,
    # 미래값을 포함하는 Hampel 필터 및 선형 보간,
    # 전체 기간 드리프트 보정을 사용할 수 있었습니다.
    #
    # 학습용 정제에서는 causal=True를 적용해
    # 미래 센서값 참조를 방지합니다.

    clean, log, rep = run_pipeline(
        raw,
        verbose=True,
        causal=True,
    )

    # 시간축 재색인으로 추가된 가상 결측 행은
    # 실제 관측값이 아니므로 학습에서 제외합니다.
    clean = clean[~clean["is_gap"].astype(bool)].copy()

    # --------------------------------------------------
    # 3. 목표변수 및 파생변수 생성
    # --------------------------------------------------

    # 향후 30분 내 고장 여부를 목표변수로 생성합니다.
    # 미래 30분을 확인할 수 없는 행은
    # 정답이 불확실하므로 학습·평가에서 제외합니다.

    d = F.make_horizon_label(
        clean,
        horizon=HORIZON,
    )

    # 롤링·차분 피처는 과거 관측값을 기준으로 생성합니다.
    d = F.build(
        d,
        windows=(10, 30, 60),
        shift_one=True,
    )

    # 목표변수와 현재 고장 여부는 입력 피처에서 제외합니다.
    # 현재 고장 여부는 에피소드 평가와 단순 이력 규칙에만 사용합니다.
    feat = [c for c in F.feature_columns(d) if c not in ("y", "machine_failure")]

    d = (
        d.dropna(subset=feat + ["y"])
        .sort_values(
            ["ts", "machine_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(d) < 500 or d["y"].nunique() < 2:
        raise SystemExit(
            f"학습에 쓸 데이터가 부족합니다({len(d)}행). 수집을 더 하세요."
        )

    # --------------------------------------------------
    # 4. 단순 이력 기반 기준 모델
    # --------------------------------------------------

    # 현재 시점의 고장 여부를 사용하지 않고
    # 설비별 이전 30개 관측치로 위험 점수를 계산합니다.
    #
    # 과거 이력은 분할 전에 계산하되,
    # 현재 또는 미래의 고장 여부는 사용하지 않습니다.
    #
    # 따라서 검증·테스트 시작 시점에도
    # 이전 구간의 과거 이력을 활용할 수 있습니다.

    naive_scores = make_naive_score(d)

    d["naive_score"] = naive_scores

    # naive_score는 기준 모델 평가에만 사용합니다.
    # RandomForest의 입력 피처에는 포함하지 않습니다.

    # --------------------------------------------------
    # 5. 학습·검증·테스트 데이터 분할
    # --------------------------------------------------

    # [수정]
    # 기존 75:25 행 단위 분할에서는
    # 같은 시각의 여러 설비가 양쪽에 나뉠 수 있었습니다.
    #
    # 고유 타임스탬프를 기준으로 60:20:20 분할하고
    # 향후 30분 정답이 다음 구간과 겹치는 행을 제외합니다.

    train, valid, test = split_by_time(
        d,
        horizon_min=HORIZON,
    )

    X_train = train[feat].to_numpy()
    y_train = train["y"].to_numpy(dtype=int)

    X_valid = valid[feat].to_numpy()
    y_valid = valid["y"].to_numpy(dtype=int)

    X_test = test[feat].to_numpy()
    y_test = test["y"].to_numpy(dtype=int)

    if len(np.unique(y_train)) < 2:
        raise SystemExit("학습 구간에 정상·고장 양쪽 클래스가 필요합니다.")

    if len(np.unique(y_valid)) < 2:
        raise SystemExit(
            "검증 구간에 정상·고장 양쪽 클래스가 필요합니다. "
            "수집 기간이나 분할 경계를 확인하세요."
        )

    print("\n[시간순 데이터 분할]")

    print(f"  train {len(train):,}행 | 양성률 {y_train.mean() * 100:.2f}%")

    print(f"  valid {len(valid):,}행 | 양성률 {y_valid.mean() * 100:.2f}%")

    print(f"  test  {len(test):,}행 | 양성률 {y_test.mean() * 100:.2f}%")

    print(f"  train 종료: {train['ts'].max()}")

    print(f"  valid 시작: {valid['ts'].min()}")

    print(f"  valid 종료: {valid['ts'].max()}")

    print(f"  test 시작:  {test['ts'].min()}")

    # 분할 경계에서 향후 30분 정답이 겹치지 않는지 확인
    embargo = pd.Timedelta(minutes=HORIZON)

    assert train["ts"].max() + embargo < valid["ts"].min(), (
        "학습·검증 경계에서 정답 구간이 겹칩니다."
    )

    assert valid["ts"].max() + embargo < test["ts"].min(), (
        "검증·테스트 경계에서 정답 구간이 겹칩니다."
    )

    # --------------------------------------------------
    # 6. RandomForest 학습 및 예측
    # --------------------------------------------------

    # 모델 학습에는 학습 구간만 사용합니다.
    # 검증 데이터는 임계값 선정에 사용하고,
    # 테스트 데이터는 최종 성능 평가에만 사용합니다.

    mdl = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    mdl.fit(
        X_train,
        y_train,
    )

    p_valid = mdl.predict_proba(X_valid)[:, 1]
    p_test = mdl.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 7. 기준 모델과 RandomForest AP 비교
    # --------------------------------------------------

    # 모든 AP를 동일한 테스트 데이터에서 계산합니다.
    #
    # 테스트 AP는 최종 평가 결과일 뿐,
    # 모델이나 임계값을 선택하는 데 사용하지 않습니다.

    naive_test = test["naive_score"].to_numpy()

    ap_baseline = float(y_test.mean())

    ap_naive = float(
        average_precision_score(
            y_test,
            naive_test,
        )
    )

    ap_rf = float(
        average_precision_score(
            y_test,
            p_test,
        )
    )

    ap_improvement = ap_rf - ap_naive

    print("\n[AP 비교: 동일 테스트 데이터]")

    print(f"  양성률 기준선          {ap_baseline:.4f}")
    print(f"  단순 이력 규칙         {ap_naive:.4f}")
    print(f"  RandomForest          {ap_rf:.4f}")
    print(f"  RF - 단순 규칙         {ap_improvement:+.4f}")

    # --------------------------------------------------
    # 8. 검증 데이터 기반 비용 최소 임계값 계산
    # --------------------------------------------------

    # 기존 결과와 비교하기 위해 행 단위 비용을 유지합니다.
    # 동일한 고장 사건을 여러 행에서 계산할 수 있으므로
    # 실제 운영 비용으로 해석하지 않습니다.
    #
    # [수정]
    # 기존에는 테스트 데이터의 정답으로
    # 최적 임계값을 선택해 평가 데이터가 누수되었습니다.
    #
    # 이제 검증 데이터에서 임계값을 한 번 선택한 뒤,
    # 테스트 데이터에서는 변경하지 않습니다.

    th, valid_cost = choose_threshold(
        y_valid,
        p_valid,
    )

    print("\n[검증 데이터 기반 임계값 선정]")

    print(f"  선택 임계값             {th:.3f}")
    print(f"  검증 행 단위 비용       {valid_cost:,}원")

    # 테스트 비용은 선택된 임계값으로만 계산합니다.
    # 테스트 비용이 더 낮은 다른 임계값을 탐색하지 않습니다.

    test_cost = calculate_row_cost(
        y_test,
        p_test,
        th,
    )

    test_cost_05 = calculate_row_cost(
        y_test,
        p_test,
        0.5,
    )

    # --------------------------------------------------
    # 9. 테스트 예측 결과 구성
    # --------------------------------------------------

    # y: 향후 30분 내 고장 여부
    # machine_failure: 현재 시점의 실제 고장 여부
    #
    # 에피소드 평가는 실제 고장 발생 시각을
    # 확인해야 하므로 machine_failure가 필요합니다.
    #
    # 기존 대시보드와의 호환성을 위해
    # 테스트 예측 결과의 컬럼 구성을 유지합니다.

    test_predictions = pd.DataFrame(
        {
            "ts": test["ts"].values,
            "machine_id": test["machine_id"].values,
            "y": y_test,
            "machine_failure": test["machine_failure"].values,
            "prob": p_test,
        }
    )

    # --------------------------------------------------
    # 10. 고장 에피소드 단위 평가
    # --------------------------------------------------

    # 실제 고장 발생 시각을 기준으로 사건을 구분하고
    # 10분 이내 반복 경고를 하나의 알람으로 묶습니다.
    #
    # [수정]
    # 검증 데이터에서 선정한 임계값을 그대로 적용해
    # 테스트 구간의 에피소드 성능을 평가합니다.
    #
    # 임계값 0.5 결과도 비교용으로 유지합니다.

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
        # [추가] 시간순 3구간 분할 정보
        "split_method": "time_based_60_20_20",
        "embargo_min": HORIZON,
        "causal_preprocessing": True,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "train_end": str(train["ts"].max()),
        "valid_start": str(valid["ts"].min()),
        "valid_end": str(valid["ts"].max()),
        "test_start": str(test["ts"].min()),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_valid": round(float(y_valid.mean()), 4),
        "positive_rate_test": round(ap_baseline, 4),
        # 최종 테스트 성능
        "roc_auc": safe_roc_auc(
            y_test,
            p_test,
        ),
        "pr_auc": round(ap_rf, 4),
        "pr_auc_naive": round(ap_naive, 4),
        "ap_improvement_over_naive": round(
            ap_improvement,
            4,
        ),
        "naive_method": ("previous_30_observations_failure_rate"),
        # [수정] 임계값 선정 데이터 출처 명시
        "threshold_selection": "validation_row_cost",
        "threshold": round(th, 3),
        "validation_cost_at_threshold": valid_cost,
        "precision": round(
            float(
                precision_score(
                    y_test,
                    (p_test >= th).astype(int),
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    (p_test >= th).astype(int),
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    (p_test >= th).astype(int),
                    zero_division=0,
                )
            ),
            4,
        ),
        # 기존 행 단위 비용
        # 테스트 구간에서 임계값을 재선정하지 않습니다.
        "cost_at_threshold": test_cost,
        "cost_at_0.5": test_cost_05,
        # 에피소드 단위 평가 결과
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

# episode_metrics()를 두 번 호출해 선택 임계값과 기본 임계값 0.5를 각각 평가함
# 기존 행 단위 비용은 cost_at_threshold와 cost_at_0.5에 유지함
# 에피소드 평가 결과는 episode_evaluation과 episode_evaluation_at_0.5에 별도로 저장함
# 두 평가 방식의 결과를 metrics.json에서 구분해 확인할 수 있도록 구현함

# 학습용 정제에 causal=True를 적용해 미래 센서값 참조를 제한함
# 기존 시간순 75:25 행 분할을 고유 시각 기준 60:20:20 분할로 변경함
# 학습·검증 경계에서 향후 30분 정답이 다음 구간과 겹치는 행을 제외함
# 검증 데이터의 행 단위 비용을 기준으로 예측 임계값을 선정함
# 테스트 데이터에서는 임계값을 변경하지 않고 최종 성능만 평가함
# metrics.json에 분할 방식, 검증 구간, 임계값 선정 출처를 추가함

# 추후 보완 사항
# 1. 전처리 중 학습 구간에서 추정한 통계의 고정 적용 방식 검토
# 2. 테스트 경계의 고장 이력 부족 문제 보완
# 3. 고장 에피소드와 독립 알람의 연결 기준 검증
# 4. 고장 에피소드 비용을 직접 최소화하는 검증 임계값 선정 방식 비교
