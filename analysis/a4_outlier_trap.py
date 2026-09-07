# python analysis\a4_outlier_trap.py

"""
A4. ★★ 센서 오작동 vs 진짜 이상 — 이상치를 지우면 안 되는 경우
===============================================================
"전처리 = 이상치 제거"라고 배웠다면 제조 데이터에서 크게 다칩니다.
설비 고장은 '이상치의 모습'으로 나타나기 때문입니다.

지우면 무슨 일이 생기는지 실데이터(AI4I)로 재고,
시뮬레이터(참값 보유)로 '왜 구분이 불가능한지'까지 확인합니다.
"""

from _common import ROOT, RAW, rule, save
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 170)
SEED = 42

# ======================================================================
rule("A4-1. AI4I에 '교과서적 이상치 제거'를 적용해 봅니다")
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
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def iqr_outlier(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return ~s.between(q1 - k * iqr, q3 + k * iqr)


flags = pd.DataFrame({c: iqr_outlier(df[c]) for c in NUM})
rows = []
for c in NUM:
    f = flags[c]
    rows.append(
        {
            "센서": c,
            "이상치": int(f.sum()),
            "이상치비율%": round(f.mean() * 100, 2),
            "이상치중 고장%": round(df.loc[f, "failure"].mean() * 100, 2)
            if f.sum()
            else 0.0,
            "정상치중 고장%": round(df.loc[~f, "failure"].mean() * 100, 2),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
base_rate = df["failure"].mean() * 100
print(f"\n전체 고장률 {base_rate:.2f}%")
print("""
★★ 읽으세요. 'IQR 이상치로 걸린 행'의 고장률이 전체 고장률보다 훨씬 높습니다.
   이상치가 곧 고장 신호입니다. 이걸 지우는 건 정답지를 태우는 겁니다.""")

any_out = flags.any(axis=1)
print(
    f"\n한 개 이상 센서에서 이상치인 행 : {any_out.sum():,}건 ({any_out.mean() * 100:.2f}%)"
)
print(f"  그중 고장 행                  : {int(df.loc[any_out, 'failure'].sum())}건")
print(
    f"  전체 고장 {int(df['failure'].sum())}건 중 "
    f"{df.loc[any_out, 'failure'].sum() / df['failure'].sum() * 100:.1f}%가 여기 들어 있습니다."
)

print("\n[고장 모드별로 — 어떤 고장이 지워지나]")
mr = []
for m in MODES:
    sub = df[df[m] == 1]
    mr.append(
        {
            "모드": m,
            "건수": len(sub),
            "이상치로 걸림": int(any_out[sub.index].sum()),
            "삭제될 비율%": round(any_out[sub.index].mean() * 100, 1),
        }
    )
print(pd.DataFrame(mr).to_string(index=False))

# ----------------------------------------------------------------------
rule("A4-2. 실제로 지우고 학습해 봅니다 (test는 원본 그대로)")
X = pd.get_dummies(df[NUM + ["type"]], columns=["type"], drop_first=True)
X["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
X["power_w"] = df["torque_nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60
X["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]
y = df["failure"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)


def fit_eval(Xa, ya, Xb, yb, name):
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xa, ya)
    p = m.predict_proba(Xb)[:, 1]
    pr = (p >= 0.5).astype(int)
    return {
        "처리": name,
        "train행": len(Xa),
        "train고장": int(ya.sum()),
        "정밀도": round(precision_score(yb, pr, zero_division=0), 4),
        "재현율": round(recall_score(yb, pr, zero_division=0), 4),
        "F1": round(f1_score(yb, pr, zero_division=0), 4),
        "PR-AUC": round(average_precision_score(yb, p), 4),
    }, p


res = []
r0, p0 = fit_eval(Xtr, ytr, Xte, yte, "① 아무것도 안 지움")
res.append(r0)

keep = ~any_out.loc[Xtr.index]
r1, p1 = fit_eval(Xtr[keep], ytr[keep], Xte, yte, "② IQR 이상치 행 삭제")
res.append(r1)

# ③ 지우지 않고 '이상치 플래그'를 피처로 추가
Xtr2 = Xtr.copy()
Xte2 = Xte.copy()
for c in NUM:
    Xtr2[f"out_{c}"] = flags[c].loc[Xtr.index].astype(int)
    Xte2[f"out_{c}"] = flags[c].loc[Xte.index].astype(int)
r2, p2 = fit_eval(Xtr2, ytr, Xte2, yte, "③ 지우지 않고 플래그로 추가")
res.append(r2)

print(pd.DataFrame(res).to_string(index=False))
print(f"""
★★★ ② 이상치를 지웠더니 train 고장이 {int(ytr.sum())}건 → {r1["train고장"]}건으로 줄고,
     재현율이 {r0["재현율"]:.4f} → {r1["재현율"]:.4f}로 떨어졌습니다.
     "데이터를 깨끗하게 만들었다"고 생각한 그 작업이 모델을 망가뜨린 겁니다.

     ③ 지우지 않고 '이상치 여부'를 피처로 넣는 게 훨씬 낫습니다.
     정보를 버리지 않으면서 모델에게 '여기 이상하다'고 알려주는 방법입니다.""")

print("\n[② 모델의 혼동행렬 — 고장을 얼마나 놓쳤나]")
print(
    pd.DataFrame(
        confusion_matrix(yte, (p1 >= 0.5).astype(int)),
        index=["실제정상", "실제고장"],
        columns=["예측정상", "예측고장"],
    ).to_string()
)

# ----------------------------------------------------------------------
rule("A4-3. ★ 시뮬레이터로 검증 — 오작동과 진짜 이상은 구분이 되나")
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from simulator import simulate_truth, pollute  # noqa: E402
from clean import hampel_flag  # noqa: E402

truth = simulate_truth(n_minutes=1440 * 14, start="2024-01-01", seed=SEED)
obs, masks = pollute(truth, seed=7, return_masks=True)
o = obs.copy()
for c in ["torque_nm", "vibration_mms", "rot_speed_rpm", "process_temp_k"]:
    o[c] = pd.to_numeric(o[c], errors="coerce")
o["ts"] = pd.to_datetime(o["ts"])
o = o.sort_values(["machine_id", "ts"])
masks = masks.loc[o.index]

print("두 종류의 '튀는 값'이 섞여 있습니다.")
print(
    f"  (A) 센서 오작동 = 우리가 주입한 스파이크 : {int(masks['spike_torque_nm'].sum())}건 (토크 기준)"
)
print(
    f"  (B) 진짜 설비 이상 = 고장 라벨          : {int(o['machine_failure'].sum())}건"
)

# IQR로 잡아 보기
out_iqr = iqr_outlier(o["torque_nm"]).fillna(False).astype(bool)
kind = pd.Series(
    np.where(
        masks["spike_torque_nm"],
        "센서오작동",
        np.where(o["machine_failure"] == 1, "진짜고장", "정상"),
    ),
    index=o.index,
    name="실제정체",
)
tab = pd.crosstab(out_iqr.rename("IQR이상치"), kind)
print("\n[토크 IQR 이상치 판정 vs 실제 정체]")
print(tab.to_string())
hit_spike = int(tab.loc[True].get("센서오작동", 0)) if True in tab.index else 0
hit_fail = int(tab.loc[True].get("진짜고장", 0)) if True in tab.index else 0
miss_fail = int(tab.loc[False].get("진짜고장", 0)) if False in tab.index else 0
tot_flag = int(out_iqr.sum())
print(f"""
★★ AI4I와 정반대 결과가 나왔습니다. 솔직히 그대로 싣습니다.
   - AI4I     : 토크 IQR 이상치의 89.9%가 진짜 고장이었습니다
   - 시뮬레이터: IQR 이상치 {tot_flag}건 중 센서오작동 {hit_spike}건
                ({hit_spike / tot_flag * 100:.1f}%), 진짜 고장은 {hit_fail}건({hit_fail / tot_flag * 100:.1f}%)뿐입니다.
                정작 고장 {miss_fail}건은 IQR 경계 '안쪽'에 남아 있습니다
                (전체 고장의 {miss_fail / (miss_fail + hit_fail) * 100:.1f}%).

   왜 갈렸나: 시뮬레이터에는 8~40배로 튀는 센서 오작동이 섞여 있습니다.
   이 극단값들이 사분위수를 밀어올려 IQR 울타리를 훨씬 바깥에 세웁니다.
   그 결과 '진짜 고장(정상보다 조금 높은 토크)'은 울타리 안에 들어와 버립니다.

   ★★★ 이게 이 절의 진짜 교훈입니다.
      IQR이 무엇을 잡을지는 '꼬리에 무엇이 들어 있느냐'에 전적으로 달려 있습니다.
      같은 규칙이 한 데이터에선 고장을 지우고, 다른 데이터에선 고장을 놓칩니다.
      → 이상치 규칙은 돌리기 전에 '무엇이 걸리는지' 반드시 눈으로 확인해야 합니다.""")

# ----------------------------------------------------------------------
rule("A4-4. 그럼 어떻게 구분하나 — 지속시간과 다른 센서와의 정합성")
o2 = o.dropna(subset=["torque_nm", "rot_speed_rpm", "vibration_mms"]).copy()
mk = masks.loc[o2.index]
grp = o2.groupby("machine_id")

# 단서 1: 스파이크는 1분짜리, 진짜 이상은 여러 분 지속됩니다
o2["tq_ham"] = grp["torque_nm"].transform(lambda s: hampel_flag(s, 11, 4.0))
o2["run"] = grp["tq_ham"].transform(
    lambda s: s.groupby((s != s.shift()).cumsum()).transform("size")
)
o2["persist"] = o2["tq_ham"] & (o2["run"] >= 3)

# 단서 2: 진짜 부하 이상이면 전류도 같이 움직입니다 (물리 정합성)
o2["cur_ham"] = grp["current_a"].transform(
    lambda s: hampel_flag(pd.to_numeric(s, errors="coerce"), 11, 4.0)
)
o2["coherent"] = o2["tq_ham"] & o2["cur_ham"]

check = []
for name, pred in [
    ("Hampel 단독", o2["tq_ham"]),
    ("지속 3분 이상", o2["persist"]),
    ("전류와 동시 이상", o2["coherent"]),
]:
    pred = pred.fillna(False).astype(bool)
    is_spike = mk["spike_torque_nm"].astype(bool)
    is_fail = o2["machine_failure"].astype(bool) & ~is_spike
    check.append(
        {
            "판정 규칙": name,
            "걸린 행": int(pred.sum()),
            "센서오작동 포함": int((pred & is_spike).sum()),
            "진짜고장 포함": int((pred & is_fail).sum()),
            "오작동 비중%": round(
                (pred & is_spike).sum() / max(pred.sum(), 1) * 100, 1
            ),
        }
    )
chk = pd.DataFrame(check)
print(chk.to_string(index=False))
r_ham = chk.iloc[0]
r_per = chk.iloc[1]
r_coh = chk.iloc[2]
print(f"""
★ 규칙별로 읽습니다.

 · Hampel 단독      : {int(r_ham["걸린 행"])}행을 잡았는데 {r_ham["오작동 비중%"]}%가 센서오작동입니다.
                     급변만 보면 오작동이 섞여 들어옵니다.

 · 지속 3분 이상     : {int(r_per["걸린 행"])}행. 기대와 달리 거의 안 잡혔습니다. 솔직히 씁니다.
                     이유: 이 시뮬레이터의 고장(야간 저부하 PWF 등)은 '급변'이 아니라
                     서서히 진입합니다. 그래서 Hampel 자체가 안 걸립니다.
                     → 지속시간 규칙은 '급변 후 지속되는 고장'에만 유효합니다.

 · 전류와 동시 이상  : {int(r_coh["걸린 행"])}행 중 센서오작동은 {int(r_coh["센서오작동 포함"])}건뿐
                     (오작동 비중 {r_coh["오작동 비중%"]}%). Hampel 단독의 {r_ham["오작동 비중%"]}%에서 크게 낮아졌습니다.
                     토크만 튀고 전류가 멀쩡하면 물리적으로 불가능 → 센서 문제입니다.
                     둘이 같이 튀면 실제로 부하가 변한 겁니다.

★★ 이게 제조 데이터 전처리의 핵심 기술입니다.
   "통계로 이상치를 찾는다"가 아니라 "여러 센서가 물리적으로 정합한가를 본다".
   센서 하나만 보는 이상치 규칙은 오작동과 고장을 절대 구분하지 못합니다.""")

# ----------------------------------------------------------------------
rule("A4-5. 그림")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ok = df[df["failure"] == 0]
ng = df[df["failure"] == 1]
ax.scatter(
    ok["tool_wear_min"], ok["torque_nm"], s=4, alpha=0.25, color="#a0aec0", label="정상"
)
ax.scatter(
    ng["tool_wear_min"],
    ng["torque_nm"],
    s=12,
    alpha=0.85,
    color="#c53030",
    label="고장",
)
q1, q3 = df["torque_nm"].quantile([0.25, 0.75])
iqr = q3 - q1
ax.axhline(q3 + 1.5 * iqr, color="black", ls="--", lw=1)
ax.axhline(q1 - 1.5 * iqr, color="black", ls="--", lw=1, label="IQR 경계")
ax.set_xlabel("공구 마모(분)")
ax.set_ylabel("토크(Nm)")
ax.set_title("AI4I — 고장은 IQR 경계 밖에 몰려 있습니다")
ax.legend(fontsize=8)

ax = axes[1]
lbl = ["① 그대로", "② IQR 삭제", "③ 플래그 추가"]
vals = [r["재현율"] for r in res]
b = ax.bar(lbl, vals, color=["#2b6cb0", "#c53030", "#276749"], width=0.6)
ax.bar_label(b, fmt="%.3f", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("재현율")
ax.set_title("이상치 처리 방식별 재현율 (AI4I)")
ax.tick_params(axis="x", labelsize=9)

ax = axes[2]
cm = pd.DataFrame(check).set_index("판정 규칙")[["센서오작동 포함", "진짜고장 포함"]]
cm.index = ["Hampel\n단독", "지속\n3분+", "전류와\n동시"]
cm.plot(kind="bar", ax=ax, color=["#c53030", "#2b6cb0"], rot=0, width=0.7)
ax.set_title("판정 규칙별 — 무엇이 걸리나 (시뮬레이터)")
ax.set_ylabel("행 수")
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "outlier_trap")


# ======================================================================
# 실행 결과 정리
# ======================================================================
# AI4I에서는 일부 IQR 이상치가 실제 고장 신호와 강하게 연결되어 있음을 확인함
# 이상치 행을 삭제하지 않고 train 고장 수와 재현율이 함께 감소해 단순 제거가 모델 성능을 떨어뜨림
# 이상치를 지우지 않고 플래그로 추가하면 정보를 유지하면서 이상 여부를 활용할 수 있음
# 시뮬레이터에서는 같은 IQR 규칙이 센서 오작동은 잡지만 진짜 고장은 대부분 놓치는 결과가 나타남
# 따라서 이상치 기준은 데이터마다 실제로 무엇이 걸리는지 확인한 뒤 사용해야 함
# 센서 하나의 통계적 이상보다 여러 센서가 물리적으로 함께 움직이는지 확인하는 것이 중요
