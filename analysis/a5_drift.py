# python analysis\a5_drift.py

"""
A5. 센서 드리프트 — 학습 시점과 운영 시점의 분포가 달라진다
============================================================
센서는 늙습니다. 열화·오염·재보정으로 값이 서서히 밀립니다.
모델은 '옛날 분포'를 배웠는데 운영은 '지금 분포'에서 돌아갑니다.

여기서는
  (1) 드리프트가 있는지 탐지하고
  (2) 있을 때 모델 성능이 얼마나 떨어지는지 재고
  (3) 보정하면 얼마나 회복되는지 확인합니다.
"""

from _common import ROOT, rule, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import features as F
from simulator import simulate_truth, pollute, POLLUTION
from clean import run_pipeline, estimate_drift, correct_drift

pd.set_option("display.width", 170)
SEED = 42
HORIZON = 30
DAYS = 14


def prep(df, horizon=HORIZON):
    df = F.make_horizon_label(df, horizon=horizon)
    df = F.build(df, windows=(10, 30, 60), shift_one=True)
    feat = [c for c in F.feature_columns(df) if c not in ("y", "machine_failure")]
    d = df.dropna(subset=feat + ["y"]).sort_values("ts").reset_index(drop=True)
    return d, feat


def split_eval(d, feat, frac=0.75, seed=SEED):
    n = len(d)
    cut = int(n * frac)
    X = d[feat].values
    y = d["y"].values.astype(int)
    m = RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2
    ).fit(X[:cut], y[:cut])
    p = m.predict_proba(X[cut:])[:, 1]
    yt = y[cut:]
    return (
        {
            "ROC-AUC": round(roc_auc_score(yt, p), 4),
            "PR-AUC": round(average_precision_score(yt, p), 4),
            "F1@0.5": round(f1_score(yt, (p >= 0.5).astype(int), zero_division=0), 4),
        },
        m,
        p,
    )


# ======================================================================
rule("A5-1. 드리프트는 눈으로 잘 안 보입니다")
truth = simulate_truth(n_minutes=1440 * DAYS, start="2024-01-01", seed=SEED)
obs = pollute(truth, seed=7)
print(f"주입한 드리프트: CNC-02 공정온도 센서 {POLLUTION['drift_per_day']} K/day")
print(f"  → {DAYS}일이면 총 {POLLUTION['drift_per_day'] * DAYS:.1f} K 밀립니다.")
print(
    f"  참고: 공정온도의 자연 표준편차는 {truth['process_temp_k'].std():.2f} K입니다."
)
print("""
★ 하루에 0.35 K씩입니다. 하루치만 보면 노이즈에 묻혀 절대 안 보입니다.
  2주가 쌓여야 4.9 K — 그제서야 '뭔가 이상한데?'가 됩니다.
  그래서 드리프트는 '긴 구간을 모아 봐야' 보입니다. 일별 집계가 필수입니다.""")

clean_corr, log, rep0 = run_pipeline(
    obs, verbose=False
)  # 파이프라인은 한 번만 돌립니다

# 보정을 되돌려 '보정 안 한 버전'을 만듭니다 (동일 조건 비교용)
t0 = clean_corr["ts"].min()
days_all = (clean_corr["ts"] - t0).dt.total_seconds() / 86400.0
uncorr = clean_corr.copy()
for mid, s in rep0["drift_applied"].items():
    if s:
        mask = uncorr["machine_id"] == mid
        uncorr.loc[mask, "process_temp_k"] = (
            uncorr.loc[mask, "process_temp_k"] + s * days_all[mask]
        )

# ----------------------------------------------------------------------
rule("A5-2. 탐지 — 일별 집계 + 회귀 기울기 검정")
d = uncorr.dropna(subset=["process_temp_k", "air_temp_k"]).copy()
d["diff"] = d["process_temp_k"] - d["air_temp_k"]
d["day"] = ((d["ts"] - d["ts"].min()).dt.total_seconds() / 86400.0).astype(int)
daily = d.groupby(["machine_id", "day"])["diff"].median().reset_index()
print("[설비별 일별 (공정온도 - 공기온도) 중앙값]")
piv = daily.pivot(index="day", columns="machine_id", values="diff").round(2)
print(piv.to_string())

print("\n[선형회귀 기울기 검정]")
rows = []
for mid, g in daily.groupby("machine_id"):
    lr = stats.linregress(g["day"], g["diff"])
    rows.append(
        {
            "설비": mid,
            "기울기(K/day)": round(lr.slope, 4),
            "표준오차": round(lr.stderr, 4),
            "t": round(lr.slope / lr.stderr, 2),
            "p값": f"{lr.pvalue:.2e}",
            "R²": round(lr.rvalue**2, 3),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
print(f"""
★ CNC-02만 기울기가 유의하게 0이 아닙니다. 나머지 둘은 0 근처입니다.
  '한 대만 밀린다' = 공정 변화가 아니라 센서 문제라는 근거입니다.
  (공정이 변했다면 세 대가 같이 움직였을 겁니다)

★ 라인 중앙값을 기준선으로 잡고 잔차의 기울기를 재면 추정값은
  {rep0["drift_slopes"]["CNC-02"]:.4f} K/day — 참값 {POLLUTION["drift_per_day"]} K/day와
  오차 {abs(rep0["drift_slopes"]["CNC-02"] - POLLUTION["drift_per_day"]) / POLLUTION["drift_per_day"] * 100:.1f}%입니다.""")

print("\n[분포 이동 검정 — 전반부 7일 vs 후반부 7일, KS 검정]")
ks_rows = []
for mid, g in uncorr.groupby("machine_id"):
    g = g.dropna(subset=["process_temp_k"])
    half = g["ts"].min() + pd.Timedelta(days=DAYS // 2)
    a = g.loc[g["ts"] < half, "process_temp_k"]
    b = g.loc[g["ts"] >= half, "process_temp_k"]
    ks = stats.ks_2samp(a, b)
    ks_rows.append(
        {
            "설비": mid,
            "전반부 평균": round(a.mean(), 2),
            "후반부 평균": round(b.mean(), 2),
            "차이": round(b.mean() - a.mean(), 2),
            "KS통계량": round(ks.statistic, 4),
            "p값": f"{ks.pvalue:.2e}",
        }
    )
print(pd.DataFrame(ks_rows).to_string(index=False))
print("""
★ 주의: KS 검정은 표본이 크면 아주 작은 차이도 p<0.001로 만듭니다.
  세 대 모두 p가 매우 작습니다. p값이 아니라 '차이의 크기'를 보세요.
  CNC-02만 차이가 실질적으로 큽니다.""")

# ----------------------------------------------------------------------
rule("A5-3. ★ 드리프트가 모델 성능에 미치는 영향")
d_un, feat = prep(uncorr)
d_co, feat2 = prep(clean_corr)
r_un, _, _ = split_eval(d_un, feat)
r_co, _, _ = split_eval(d_co, feat2)
print(
    pd.DataFrame(
        [dict(데이터="드리프트 방치", **r_un), dict(데이터="드리프트 보정", **r_co)]
    ).to_string(index=False)
)

# CNC-02만 따로
print("\n[CNC-02만 떼어서 — 드리프트가 있는 그 설비]")
sub_un = d_un[d_un["machine_id"] == "CNC-02"].reset_index(drop=True)
sub_co = d_co[d_co["machine_id"] == "CNC-02"].reset_index(drop=True)
r2_un, _, _ = split_eval(sub_un, feat)
r2_co, _, _ = split_eval(sub_co, feat2)
print(
    pd.DataFrame(
        [dict(데이터="드리프트 방치", **r2_un), dict(데이터="드리프트 보정", **r2_co)]
    ).to_string(index=False)
)
gain = (r2_co["PR-AUC"] - r2_un["PR-AUC"]) / max(r2_un["PR-AUC"], 1e-9) * 100
print(f"""
★★★ 예상과 반대 결과입니다. 그대로 싣습니다.
   드리프트를 보정했더니 모델 성능이 오히려 조금 '떨어졌습니다'.
     전체    PR-AUC {r_un["PR-AUC"]:.4f} → {r_co["PR-AUC"]:.4f}
     CNC-02  PR-AUC {r2_un["PR-AUC"]:.4f} → {r2_co["PR-AUC"]:.4f} ({gain:+.1f}%)

   왜 그런가 — 두 가지로 봅니다.
   1) 트리 모델은 '시간에 따라 단조 증가하는 편향'을 다른 피처와 조합해
      스스로 흡수합니다. 보정해 준다고 크게 득 볼 게 없습니다.
   2) 보정은 추정값(기울기 {rep0["drift_slopes"]["CNC-02"]:.4f})을 빼는 작업이라
      추정 오차만큼 새 잡음이 들어갑니다. 이득보다 잡음이 컸습니다.
   ※ 차이가 작아서 '보정이 해롭다'고 단정할 수는 없습니다.
     seed를 바꾸면 부호가 뒤집힐 수 있는 크기입니다. 그 점도 밝혀 둡니다.

★★ 그러면 드리프트 보정은 왜 하나 — 모델이 아니라 '룰'이 먼저 깨지기 때문입니다.
   대부분의 현장은 아직 고정 임계값 경보로 돌아갑니다. 다음 절을 보세요.""")

# ----------------------------------------------------------------------
rule("A5-4. 고정 임계값 경보가 드리프트에 어떻게 무너지는가")
th = 315.0
al = []
for name, frame in [("보정 안 함", uncorr), ("보정 함", clean_corr)]:
    g = frame.dropna(subset=["process_temp_k"]).copy()
    g["week"] = np.where(
        (g["ts"] - g["ts"].min()).dt.days < DAYS // 2, "1주차", "2주차"
    )
    for mid in ["CNC-01", "CNC-02", "CNC-03"]:
        s = g[g["machine_id"] == mid]
        r = s.groupby("week")["process_temp_k"].apply(lambda x: (x > th).mean() * 100)
        al.append(
            {
                "데이터": name,
                "설비": mid,
                "1주차 경보율%": round(r.get("1주차", np.nan), 2),
                "2주차 경보율%": round(r.get("2주차", np.nan), 2),
            }
        )
alarm = pd.DataFrame(al)
print(f"[공정온도 {th} K 초과 경보 발생률]")
print(alarm.to_string(index=False))
print("""
★★ 보정 안 한 CNC-02의 경보율이 2주차에 급증합니다.
   설비는 멀쩡한데 센서가 밀려서 나는 '가짜 경보'입니다.
   현장에서 이런 일이 반복되면 작업자가 경보를 꺼버립니다.
   → 그다음 진짜 고장이 나면 아무도 모릅니다. 이게 드리프트가 무서운 진짜 이유입니다.""")

# ----------------------------------------------------------------------
rule("A5-5. 그림")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for mid, g in daily.groupby("machine_id"):
    axes[0].plot(g["day"], g["diff"], marker="o", ms=4, label=mid)
    lr = stats.linregress(g["day"], g["diff"])
    axes[0].plot(g["day"], lr.intercept + lr.slope * g["day"], ls="--", lw=1, alpha=0.7)
axes[0].set_xlabel("경과 일")
axes[0].set_ylabel("공정온도 - 공기온도 (K)")
axes[0].set_title("일별 중앙값 — CNC-02만 우상향")
axes[0].legend(fontsize=8)

for mid, color in [("CNC-02", "#c53030"), ("CNC-01", "#2b6cb0")]:
    g = uncorr[uncorr["machine_id"] == mid].dropna(subset=["process_temp_k"])
    half = g["ts"].min() + pd.Timedelta(days=DAYS // 2)
    axes[1].hist(
        g.loc[g["ts"] < half, "process_temp_k"],
        bins=60,
        alpha=0.5,
        density=True,
        color=color,
        label=f"{mid} 1주차",
    )
    axes[1].hist(
        g.loc[g["ts"] >= half, "process_temp_k"],
        bins=60,
        alpha=0.5,
        density=True,
        histtype="step",
        lw=1.8,
        color=color,
        label=f"{mid} 2주차",
    )
axes[1].set_title("분포 이동 (보정 전)")
axes[1].set_xlabel("공정온도 (K)")
axes[1].legend(fontsize=7)

pv = alarm[alarm["데이터"] == "보정 안 함"].set_index("설비")[
    ["1주차 경보율%", "2주차 경보율%"]
]
pv.plot(kind="bar", ax=axes[2], rot=0, color=["#a0aec0", "#c53030"])
axes[2].set_title(f"고정 임계값({th}K) 경보율 — 보정 전")
axes[2].set_ylabel("%")
axes[2].legend(fontsize=8)
fig.tight_layout()
save(fig, "drift")


# ======================================================================
# 실행 결과 정리
# ======================================================================
# CNC-02에 주입한 0.35 K/day 드리프트를 일별 집계와 회귀 기울기로 탐지함
# 추정 기울기는 0.3464 K/day로 참값과 1.1%의 오차가 나타남
# KS 검정은 작은 차이에도 유의하게 나올 수 있어 p값과 실제 변화 크기를 함께 확인해야 함
# 드리프트 보정 후 RandomForest 성능은 오히려 소폭 감소해 보정이 항상 모델 성능을 높이는 것은 아님을 확인
# 반면 CNC-02의 315K 초과 경보율은 보정 전 2주차 65.97%, 보정 후 6.66%로 크게 차이 남
# 따라서 드리프트는 모델 성능뿐 아니라 고정 임계값 기반 가짜 경보 문제까지 함께 확인해야 함
