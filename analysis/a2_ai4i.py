"""
A2. AI4I 2020 (UCI) — ★ 정확도가 쓰레기 지표인 이유
====================================================
실데이터입니다. 10,000행, 고장 339건(3.39%).
"전부 정상"이라고만 찍어도 정확도 96.6%가 나옵니다.
"""

from _common import RAW, rule, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 170)
SEED = 42

# ----------------------------------------------------------------------
rule("A2-1. 로드 & 첫 점검")
df = pd.read_csv(RAW / "ai4i" / "ai4i2020.csv")
print("shape:", df.shape)  # shape: (10000, 14)
print("\n[컬럼]")
print(
    pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "결측": df.isna().sum(),
            "고유값": df.nunique(),
        }
    ).to_string()
)
print("\n중복 행:", df.duplicated().sum(), "| 중복 UDI:", df["UDI"].duplicated().sum())
# [컬럼]
#                            dtype  결측    고유값
# UDI                        int64   0  10000
# Product ID                   str   0  10000
# Type                         str   0      3
# Air temperature [K]      float64   0     93
# Process temperature [K]  float64   0     82
# Rotational speed [rpm]     int64   0    941
# Torque [Nm]              float64   0    577
# Tool wear [min]            int64   0    246
# Machine failure            int64   0      2
# TWF                        int64   0      2
# HDF                        int64   0      2
# PWF                        int64   0      2
# OSF                        int64   0      2
# RNF                        int64   0      2
# 중복 행: 0 | 중복 UDI: 0

# 컬럼명에 대괄호, 공백이 다수 포함되어 있으므로 먼저 정리하는 작업을 거침
REN = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
    "Product ID": "product_id",
    "Type": "type",
}
df = df.rename(columns=REN)
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUM = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
print("\n[수치형 요약]")
print(df[NUM].describe().loc[["mean", "std", "min", "50%", "max"]].round(2).to_string())
# [수치형 요약]
#       air_temp_k  process_temp_k  rot_speed_rpm  torque_nm  tool_wear_min
# mean       300.0          310.01        1538.78      39.99         107.95
# std          2.0            1.48         179.28       9.97          63.65
# min        295.3          305.70        1168.00       3.80           0.00
# 50%        300.1          310.10        1503.00      40.10         108.00
# max        304.5          313.80        2886.00      76.60         253.00
print("\n[품질 등급]")
print(df["type"].value_counts().to_string())
# [품질 등급]
# type
# L    6000
# M    2997
# H    1003

# ----------------------------------------------------------------------
rule("A2-2. ★ 라벨 검산 — 실데이터는 라벨부터 모순됩니다")
anyf = df[MODES].sum(axis=1)
bad1 = ((df["failure"] == 1) & (anyf == 0)).sum()
bad2 = ((df["failure"] == 0) & (anyf > 0)).sum()
print(f"고장=1 인데 세부 모드가 하나도 없음 : {bad1}건")
# 고장=1 인데 세부 모드가 하나도 없음 : 9건
print(f"세부 모드가 있는데 고장=0          : {bad2}건")
# 세부 모드가 있는데 고장=0          : 18건
print("\n두 번째 경우의 내역:")
print(df[(df["failure"] == 0) & (anyf > 0)][MODES].sum().to_string())
# 두 번째 경우의 내역:
# TWF     0
# HDF     0
# PWF     0
# OSF     0
# RNF    18
print("""
★ RNF(원인불명 고장) 19건 중 18건이 'Machine failure=0'으로 되어 있습니다.
  데이터 제공자의 정의상 RNF는 최종 고장 라벨에 포함되지 않은 것으로 보입니다.
  이런 건 '틀렸다'가 아니라 '정의를 확인해야 한다'입니다.
  → 이 분석에서는 원본 Machine failure 컬럼을 그대로 목표변수로 씁니다.
    대신 '라벨 정의에 이런 특이점이 있다'를 보고서 한계 항목에 적습니다.""")

print("\n[모드별 건수 / 전체 대비]")
mt = pd.DataFrame({"건수": df[MODES + ["failure"]].sum()})
mt["비율(%)"] = (mt["건수"] / len(df) * 100).round(2)
print(mt.to_string())
# [모드별 건수 / 전체 대비]
#           건수  비율(%)
# TWF       46   0.46
# HDF      115   1.15
# PWF       95   0.95
# OSF       98   0.98
# RNF       19   0.19
# failure  339   3.39

# # ----------------------------------------------------------------------
rule("A2-3. ★★★ 불균형 — 정확도가 쓰레기 지표인 이유")
rate = df["failure"].mean()
print(f"고장률          : {rate * 100:.2f}%  ({df['failure'].sum()}건 / {len(df)}건)")
# 고장률          : 3.39%  (339건 / 10000건)
print(f"정상률          : {(1 - rate) * 100:.2f}%")
# 정상률          : 96.61%
print(f"\n★ '무조건 정상'이라고 찍는 모델의 정확도 = {(1 - rate) * 100:.2f}%")
# ★ '무조건 정상'이라고 찍는 모델의 정확도 = 96.61%
print("  이 모델은 고장을 단 한 건도 못 잡습니다. 그런데 정확도는 96.61%입니다.")
X = pd.get_dummies(df[NUM + ["type"]], columns=["type"], drop_first=True)
# x : 모델에게 입력으로 줄 데이터(문제), 센서값과 제품 등급을 체크
y = df["failure"]
# y : 모델이 맞혀야 하는 정답, 고장인지 정상인지 맞힘
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)
# 25%를 테스트 데이터로 사용
# 전체 데이터의 고장률(정상=96.61%, 고장=3.39%) 데이터가 심하게 불균형
# 무작위하게 나누다보면 비율이 달라질 가능성 존재
# stratify=y로 train과 test에서도 원래의 정상/고장 비율이 비슷하게 유지되도록 함
print(f"\ntrain {Xtr.shape} / test {Xte.shape} | test 고장 {yte.sum()}건")
# train (7500, 7) / test (2500, 7) | test 고장 85건

dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
# 제대로 학습하는 모델이 아닌, 기준점 비교용 모델을 생성
dp = dummy.predict(Xte)
print("\n[무조건 정상 모델]")
print(f"  정확도  : {accuracy_score(yte, dp):.4f}")
print(f"  정밀도  : {precision_score(yte, dp, zero_division=0):.4f}")
print(f"  재현율  : {recall_score(yte, dp, zero_division=0):.4f}")
print(f"  F1      : {f1_score(yte, dp, zero_division=0):.4f}")
print("  혼동행렬:")
print(
    pd.DataFrame(
        confusion_matrix(yte, dp),
        index=["실제정상", "실제고장"],
        columns=["예측정상", "예측고장"],
    ).to_string()
)
# [무조건 정상 모델]
#   정확도  : 0.9660
#   정밀도  : 0.0000
#   재현율  : 0.0000
#   F1      : 0.0000
#   혼동행렬:
#       예측정상  예측고장
# 실제정상  2415     0
# 실제고장    85     0
print(
    f"\n★★ 정확도 {accuracy_score(yte, dp):.4f} / 재현율 0.0000. 이 한 줄이 전부입니다."
)
# 정확도 0.9660 / 재현율 0.0000.
print("   포트폴리오에 '정확도 96%'라고 쓰면 면접관은 이 표를 떠올립니다.")

#  ----------------------------------------------------------------------
rule("A2-4. 실제 모델 — 로지스틱 vs 랜덤포레스트")
models = {
    "로지스틱": Pipeline(
        [
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    ),
    "로지스틱+가중": Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "m",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    ),
    "랜덤포레스트": RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1
    ),
    "랜덤포레스트+가중": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
    ),
}
rows, probs = [], {}
for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    probs[name] = p
    pred = (p >= 0.5).astype(int)
    rows.append(
        {
            "모델": name,
            "정확도": round(accuracy_score(yte, pred), 4),
            "정밀도": round(precision_score(yte, pred, zero_division=0), 4),
            "재현율": round(recall_score(yte, pred, zero_division=0), 4),
            "F1": round(f1_score(yte, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(yte, p), 4),
            "PR-AUC": round(average_precision_score(yte, p), 4),
        }
    )
base = pd.DataFrame(rows)
base.loc[len(base)] = {
    "모델": "무조건정상",
    "정확도": round(accuracy_score(yte, dp), 4),
    "정밀도": 0.0,
    "재현율": 0.0,
    "F1": 0.0,
    "ROC-AUC": 0.5,
    "PR-AUC": round(yte.mean(), 4),
}
print(base.to_string(index=False))
#        모델    정확도    정밀도    재현율     F1  ROC-AUC  PR-AUC
#      로지스틱 0.9676 0.6000 0.1412 0.2286   0.8768  0.4051
#   로지스틱+가중 0.8252 0.1393 0.8000 0.2373   0.8836  0.3834
#    랜덤포레스트 0.9824 0.8596 0.5765 0.6901   0.9635  0.7889
# 랜덤포레스트+가중 0.9800 0.7160 0.6824 0.6988   0.9697  0.7625
#     무조건정상 0.9660 0.0000 0.0000 0.0000   0.5000  0.0340
print(f"""
★ 읽는 법
  - 정확도는 전부 0.96~0.98입니다. 모델을 구분하지 못합니다. 쓸모없는 지표입니다.
  - PR-AUC의 기준선은 '고장률' 자체입니다 = {yte.mean():.4f}.
    이것보다 얼마나 높은지가 진짜 성능입니다.
  - ROC-AUC는 불균형에서 낙관적으로 보입니다. 고장률 3%면 PR-AUC를 보세요.""")

print("\n[랜덤포레스트 상세 리포트]")
best = "랜덤포레스트"
pred = (probs[best] >= 0.5).astype(int)
print(classification_report(yte, pred, target_names=["정상", "고장"], digits=4))
print("혼동행렬:")
print(
    pd.DataFrame(
        confusion_matrix(yte, pred),
        index=["실제정상", "실제고장"],
        columns=["예측정상", "예측고장"],
    ).to_string()
)
# [랜덤포레스트 상세 리포트]
#               precision    recall  f1-score   support

#           정상     0.9853    0.9967    0.9909      2415
#           고장     0.8596    0.5765    0.6901        85

#     accuracy                         0.9824      2500
#    macro avg     0.9225    0.7866    0.8405      2500
# weighted avg     0.9810    0.9824    0.9807      2500

# 혼동행렬:
#       예측정상  예측고장
# 실제정상  2407     8
# 실제고장    36    49

# ----------------------------------------------------------------------
rule("A2-5. ROC와 PR 곡선 — 같은 모델, 다른 인상")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for name in ["로지스틱", "랜덤포레스트"]:
    fpr, tpr, _ = roc_curve(yte, probs[name])
    axes[0].plot(
        fpr, tpr, lw=1.6, label=f"{name} (AUC={roc_auc_score(yte, probs[name]):.3f})"
    )
    pr, rc, _ = precision_recall_curve(yte, probs[name])
    axes[1].plot(
        rc,
        pr,
        lw=1.6,
        label=f"{name} (AP={average_precision_score(yte, probs[name]):.3f})",
    )
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="무작위")
axes[0].set_xlabel("거짓양성률 FPR")
axes[0].set_ylabel("재현율 TPR")
axes[0].set_title("ROC 곡선 — 좋아 보입니다")
axes[0].legend(fontsize=8)
axes[1].axhline(
    yte.mean(), color="k", ls="--", lw=0.8, label=f"기준선 = 고장률 {yte.mean():.3f}"
)
axes[1].set_xlabel("재현율")
axes[1].set_ylabel("정밀도")
axes[1].set_title("PR 곡선 — 현실은 이쪽입니다")
axes[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "ai4i_roc_pr")
# 실제 경로
# C:\Users\swrkd\Desktop\predictive-maintenance\outputs\figures\pd_ai4i_roc_pr.png

# ----------------------------------------------------------------------
rule("A2-6. 고장 모드별로 나눠 보면 — 왜 어떤 고장은 못 잡나")
rf = models["랜덤포레스트"]
te = df.loc[Xte.index].copy()
te["prob"] = probs["랜덤포레스트"]
te["pred"] = (te["prob"] >= 0.5).astype(int)
mm = []
for md in MODES:
    sub = te[te[md] == 1]
    if len(sub) == 0:
        continue
    mm.append(
        {
            "모드": md,
            "test 건수": len(sub),
            "잡아낸 건수": int(sub["pred"].sum()),
            "재현율": round(sub["pred"].mean(), 3),
            "평균확률": round(sub["prob"].mean(), 3),
        }
    )
print(pd.DataFrame(mm).to_string(index=False))
#  모드  test 건수  잡아낸 건수   재현율  평균확률
# TWF       13       1 0.077 0.116
# HDF       36      21 0.583 0.516
# PWF       17      14 0.824 0.708
# OSF       20      16 0.800 0.630
# RNF        6       0 0.000 0.069
print("""
★ RNF(원인불명)의 재현율이 0에 가깝습니다. 당연합니다 — 무작위로 발생하니까
  센서에 신호가 없습니다. '모델이 못 잡는 게 아니라 잡을 수 없는 것'입니다.
  이걸 구분해서 말할 수 있으면 면접에서 확실히 다릅니다.""")

print("\n[변수 중요도 — 랜덤포레스트]")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.round(4).to_string())
# [변수 중요도 — 랜덤포레스트]
# torque_nm         0.3328
# rot_speed_rpm     0.2247
# tool_wear_min     0.1663
# air_temp_k        0.1277
# process_temp_k    0.1227
# type_L            0.0153
# type_M            0.0105

# ----------------------------------------------------------------------
rule("A2-7. 파생변수를 넣으면 — 도메인 지식의 힘")
X2 = X.copy()
X2["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
X2["power_w"] = df["torque_nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60
X2["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]
X2tr, X2te = X2.loc[Xtr.index], X2.loc[Xte.index]
rf2 = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(
    X2tr, ytr
)
p2 = rf2.predict_proba(X2te)[:, 1]
print(
    f"파생변수 없음 : PR-AUC {average_precision_score(yte, probs['랜덤포레스트']):.4f} | "
    f"F1 {f1_score(yte, (probs['랜덤포레스트'] >= 0.5).astype(int)):.4f}"
)
# 파생변수 없음 : PR-AUC 0.7889 | F1 0.6901
print(
    f"파생변수 3개  : PR-AUC {average_precision_score(yte, p2):.4f} | "
    f"F1 {f1_score(yte, (p2 >= 0.5).astype(int)):.4f}"
)
# 파생변수 3개  : PR-AUC 0.8966 | F1 0.8861
imp2 = pd.Series(rf2.feature_importances_, index=X2.columns).sort_values(
    ascending=False
)
print("\n[중요도 상위 6개]")
print(imp2.head(6).round(4).to_string())
# [중요도 상위 6개]
# wear_torque      0.2184
# power_w          0.2137
# rot_speed_rpm    0.1639
# torque_nm        0.1268
# temp_diff_k      0.1173
# tool_wear_min    0.0517
print("""
★ temp_diff / power / wear×torque 는 AI4I의 고장 정의식 그 자체입니다.
  모델을 바꾸는 것보다 '고장이 어떻게 정의되는지'를 아는 게 성능을 올립니다.
  이게 제조 도메인 지식의 값어치입니다.""")

np.save(RAW.parent / "ai4i_probs.npy", probs["랜덤포레스트"])
te[["UDI", "prob", "pred", "failure"] + MODES].to_csv(
    RAW.parent / "ai4i_test_pred.csv", index=False
)
print("\n저장: data/ai4i_test_pred.csv")


# ---------------------------------------------------------------
# 고장률이 3.39%로 불균형하기 때문에 정확도만으로 모델 성능을 판단할 수 없음
# 실제로 무조건 정상으로 예측하더라도 정확도는 96.6%이지만 고장 재현율은 0%임을 확인
# 따라서 불균형 데이터에서는 정밀도, 재현율, F1, PR-AUC를 함께 확인할 것

# 세부 고장 모드와 최종 고장 라벨이 일치하지 않는 경우가 있음
# 학습 전 라벨 정의를 먼저 확인할 것

# 고장 모드별로 탐지 성능이 다름을 확인
# 센서에 뚜렷한 시놓가 없는 RNF는 모델이 탐지하는 데에 어려움이 있었음

# 기존 센서값을 그대로 사용하는 것보다 온도차, 출력, 마모*토크 처럼 고장 원리를 반영한 파생변수를 추가
# PR-AUC가 0.7889에서 0.8966으로 향상
# F1 또한 0.6907에서 0.8861로 향상
# 모델이 고장 패턴을 더욱 잘 학습하였음
# 모델 자체를 바꾸는 것 뿐만 아니라 고장 원리를 이해하고 적절한 변수를 만들어야 함
