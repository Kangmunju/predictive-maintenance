# python analysis\a6_cost_threshold.py

"""
A6. ★★ 비용 비대칭 — 임계값을 0.5로 쓰면 안 되는 이유
=======================================================
0.5는 "두 오류의 비용이 같다"는 가정입니다.
설비에서 그런 경우는 없습니다.

  고장 놓침(FN) : 라인 정지 + 긴급 정비 + 납기 지연  → 수백~수천만원
  과잉 정비(FP) : 점검 인건비 + 짧은 계획 정지        → 수십만원

비용이 다르면 임계값도 달라야 합니다. 그걸 '숫자로' 정합니다.
"""

from _common import RAW, ROOT, rule, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 170)
SEED = 42

# 현장에서 받아오는 숫자입니다. 지어내면 안 되고, 정비팀에 물어봐야 합니다.
COST_FN = 8_000_000  # 고장 1건 놓침 = 라인 정지 2시간 + 긴급 부품
COST_FP = 300_000  # 헛点검 1건 = 정비사 2인 × 2시간 + 계획 정지

# ======================================================================
rule("A6-1. 모델 준비 (AI4I)")
df = pd.read_csv(RAW / "ai4i" / "ai4i2020.csv").rename(
    columns={
        "Air temperature [K]": "air_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rot_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "failure",
        "Type": "type",
    }
)
NUM = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
X = pd.get_dummies(df[NUM + ["type"]], columns=["type"], drop_first=True)
X["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
X["power_w"] = df["torque_nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60
X["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]
y = df["failure"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)
mdl = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(
    Xtr, ytr
)
p = mdl.predict_proba(Xte)[:, 1]
print(f"test {len(yte):,}건 | 고장 {int(yte.sum())}건 ({yte.mean() * 100:.2f}%)")
print(f"PR-AUC {average_precision_score(yte, p):.4f}")

# ----------------------------------------------------------------------
rule("A6-2. 임계값을 바꾸면 무슨 일이 생기나")
rows = []
for th in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    cost = fn * COST_FN + fp * COST_FP
    rows.append(
        {
            "임계값": th,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "정밀도": round(precision_score(yte, pred, zero_division=0), 3),
            "재현율": round(recall_score(yte, pred, zero_division=0), 3),
            "F1": round(f1_score(yte, pred, zero_division=0), 3),
            "총비용(만원)": int(cost / 10_000),
        }
    )
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

best_f1 = tab.loc[tab["F1"].idxmax()]
best_cost = tab.loc[tab["총비용(만원)"].idxmin()]
at_05 = tab[tab["임계값"] == 0.50].iloc[0]
print(f"""
★ F1이 최대인 임계값   : {best_f1["임계값"]}  (F1 {best_f1["F1"]}, 비용 {best_f1["총비용(만원)"]:,}만원)
★ 비용이 최소인 임계값 : {best_cost["임계값"]}  (F1 {best_cost["F1"]}, 비용 {best_cost["총비용(만원)"]:,}만원)
★ 관행대로 0.5를 쓰면  : F1 {at_05["F1"]}, 비용 {at_05["총비용(만원)"]:,}만원

  0.5 대비 절감액 = {at_05["총비용(만원)"] - best_cost["총비용(만원)"]:,}만원
  ({(at_05["총비용(만원)"] - best_cost["총비용(만원)"]) / max(at_05["총비용(만원)"], 1) * 100:.1f}% 절감)
  ★ F1이 최대인 지점과 비용이 최소인 지점이 다릅니다. F1은 비용을 모릅니다.""")

# ----------------------------------------------------------------------
rule("A6-3. 세밀하게 훑어서 최적 임계값 찾기")
ths = np.linspace(0.01, 0.99, 197)
costs, recs, precs, f1s = [], [], [], []
for th in ths:
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    costs.append(fn * COST_FN + fp * COST_FP)
    recs.append(recall_score(yte, pred, zero_division=0))
    precs.append(precision_score(yte, pred, zero_division=0))
    f1s.append(f1_score(yte, pred, zero_division=0))
costs = np.array(costs)
i_best = int(costs.argmin())
i_f1 = int(np.argmax(f1s))
print(
    f"비용 최소 임계값 : {ths[i_best]:.3f} → {costs[i_best] / 10_000:,.0f}만원 "
    f"(재현율 {recs[i_best]:.3f}, 정밀도 {precs[i_best]:.3f})"
)
print(
    f"F1  최대 임계값 : {ths[i_f1]:.3f} → {costs[i_f1] / 10_000:,.0f}만원 "
    f"(재현율 {recs[i_f1]:.3f}, 정밀도 {precs[i_f1]:.3f})"
)
print(f"임계값 0.5      : {costs[np.argmin(np.abs(ths - 0.5))] / 10_000:,.0f}만원")
print(
    f"전부 정비(th=0) : {int(yte.sum() == 0) * 0 + (len(yte) - yte.sum()) * COST_FP / 10_000:,.0f}만원"
)
print(f"아무것도 안 함  : {yte.sum() * COST_FN / 10_000:,.0f}만원")

# ----------------------------------------------------------------------
rule("A6-4. ★ 비용 비율이 바뀌면 임계값도 바뀝니다")
rows = []
for ratio in [1, 3, 5, 10, 26, 50, 100]:
    c_fn = COST_FP * ratio
    cc = np.array(
        [
            confusion_matrix(yte, (p >= t).astype(int), labels=[0, 1]).ravel()
            for t in ths
        ]
    )
    tot = cc[:, 2] * c_fn + cc[:, 1] * COST_FP
    i = int(tot.argmin())
    pred = (p >= ths[i]).astype(int)
    rows.append(
        {
            "FN/FP 비용비": f"{ratio}:1",
            "최적 임계값": round(ths[i], 3),
            "재현율": round(recall_score(yte, pred, zero_division=0), 3),
            "정밀도": round(precision_score(yte, pred, zero_division=0), 3),
            "놓친 고장": int(cc[i, 2]),
            "헛점검": int(cc[i, 1]),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
print(f"""
★ 고장 놓침이 비쌀수록 임계값이 내려가고 재현율이 올라갑니다. 당연한 방향입니다.
  우리 가정({COST_FN // 10_000}만원 : {COST_FP // 10_000}만원 = {COST_FN // COST_FP}:1)에서는
  임계값 {ths[i_best]:.2f}가 답입니다.

★★ 면접에서 이렇게 말하세요.
  "임계값은 모델이 정하는 게 아니라 비용 구조가 정합니다.
   저는 FN:FP 비용비를 {COST_FN // COST_FP}:1로 가정하고 총비용 최소점을 찾았고,
   비용비가 바뀌면 임계값이 어떻게 움직이는지 민감도까지 확인했습니다."
  숫자를 어디서 가져왔는지(가정인지 실측인지) 반드시 밝히세요.""")

# ----------------------------------------------------------------------
rule("A6-5. 정비 용량 제약 — 현실은 '상위 N건만 볼 수 있다'")
print("정비팀이 하루에 볼 수 있는 건수가 정해져 있는 경우입니다.")
order = np.argsort(-p)
yt = yte.values[order]
rows = []
for k in [10, 20, 30, 50, 100, 200]:
    hit = int(yt[:k].sum())
    rows.append(
        {
            "상위 K건 점검": k,
            "잡은 고장": hit,
            "Precision@K": round(hit / k, 3),
            "Recall@K": round(hit / yte.sum(), 3),
            "무작위였다면": round(k * yte.mean(), 1),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
print(f"""
★ 상위 100건만 점검해도 전체 고장 {int(yte.sum())}건 중
  {int(yt[:100].sum())}건({yt[:100].sum() / yte.sum() * 100:.0f}%)을 잡습니다.
  무작위로 100건 뽑으면 {100 * yte.mean():.1f}건입니다. 이게 모델의 실질 가치입니다.
★ 실무 보고서에는 PR-AUC보다 이 표가 훨씬 잘 먹힙니다.
  "하루 100건 점검 → 고장 {yt[:100].sum() / yte.sum() * 100:.0f}% 예방"이 경영진이 이해하는 언어입니다.""")

# ----------------------------------------------------------------------
rule("A6-6. 그림")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ax.plot(ths, costs / 10_000, color="#c53030", lw=1.8)
ax.axvline(ths[i_best], color="#2b6cb0", ls="--", label=f"비용최소 {ths[i_best]:.2f}")
ax.axvline(0.5, color="gray", ls=":", label="관행 0.5")
ax.set_xlabel("임계값")
ax.set_ylabel("총비용 (만원)")
ax.set_title(f"총비용 곡선 (FN:FP = {COST_FN // COST_FP}:1)")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(ths, recs, label="재현율", color="#2b6cb0")
ax.plot(ths, precs, label="정밀도", color="#c53030")
ax.plot(ths, f1s, label="F1", color="#276749", ls="--")
ax.axvline(ths[i_best], color="black", ls="--", lw=1)
ax.set_xlabel("임계값")
ax.set_title("임계값에 따른 지표 변화")
ax.legend(fontsize=8)

ax = axes[2]
ks = np.arange(1, 301)
hits = np.cumsum(yt[:300])
ax.plot(ks, hits / yte.sum(), color="#2b6cb0", lw=1.8, label="모델 순위")
ax.plot(ks, ks * yte.mean() / yte.sum(), color="gray", ls="--", label="무작위")
ax.set_xlabel("점검 건수 K")
ax.set_ylabel("잡은 고장 비율")
ax.set_title("정비 용량 제약 하의 성능")
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "cost_threshold")

pd.DataFrame(
    {"threshold": ths, "cost_won": costs, "recall": recs, "precision": precs, "f1": f1s}
).to_csv(ROOT / "data" / "threshold_curve.csv", index=False)
print("저장: data/threshold_curve.csv")


# ======================================================================
# 실행 결과 정리
# ======================================================================
# 임계값 0.5에서는 F1이 높았지만 비용 최소 임계값과는 차이가 있음을 확인함
# FN 800만원, FP 30만원으로 가정했을 때 비용 최소 임계값은 0.06으로 나타남
# 임계값 0.5의 총비용 12,090만원보다 0.06에서 7,770만원으로 낮아짐
# 고장 놓침 비용이 커질수록 최적 임계값이 낮아지고 재현율이 높아지는 것을 확인함
# 따라서 운영 임계값은 고정된 0.5가 아니라 실제 FN과 FP의 비용 구조를 기준으로 결정해야 함
# 위험도 상위 100건만 점검해도 실제 고장 85건 중 72건을 포획함
