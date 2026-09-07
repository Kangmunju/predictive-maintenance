# python analysis\a7_secom.py

"""
A7. SECOM (UCI) — 진짜 지저분한 반도체 공정 데이터
==================================================
590개 센서 × 1,567개 웨이퍼 로트. 불량률 6.6%.
결측 대량, 상수 컬럼 대량, 고차원, 극도 불균형.

★ 이 절의 결론은 "성능이 안 나옵니다"입니다. 그걸 그대로 씁니다.
  안 나오는 이유를 설명할 수 있는 것이 억지로 올린 숫자보다 낫습니다.
"""

from _common import RAW, ROOT, rule, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 170)
SEED = 42

# ======================================================================
rule("A7-1. 로드 — 컬럼 이름조차 없습니다")
X = pd.read_csv(RAW / "secom" / "secom.data", sep=" ", header=None)
X.columns = [f"s{i:03d}" for i in range(X.shape[1])]
lab = pd.read_csv(
    RAW / "secom" / "secom_labels.data", sep=" ", header=None, names=["label", "ts"]
)
lab["ts"] = pd.to_datetime(lab["ts"].str.strip('"'), format="%d/%m/%Y %H:%M:%S")
print("센서 행렬:", X.shape, "| 라벨:", lab.shape)
print(
    "기간:",
    lab["ts"].min(),
    "~",
    lab["ts"].max(),
    f"({(lab['ts'].max() - lab['ts'].min()).days}일)",
)
y = (lab["label"] == 1).astype(int)  # 1 = 불량(fail)
print(
    f"\n불량 {int(y.sum())}건 / 정상 {int((1 - y).sum())}건 = 불량률 {y.mean() * 100:.3f}%"
)
print("""
★ 타임스탬프 형식이 DD/MM/YYYY입니다. pandas 기본값(MM/DD)으로 읽으면
  19/07을 '19월 7일'로 해석해 통째로 NaT가 되거나, 07/08 같은 날짜가
  조용히 뒤바뀝니다. format을 반드시 명시하세요.""")

# ----------------------------------------------------------------------
rule("A7-2. 데이터 품질 점검 — 손도 못 댈 상태입니다")
miss = X.isna().mean()
nuniq = X.nunique(dropna=True)
print(
    f"전체 결측: {int(X.isna().sum().sum()):,}개 ({X.isna().sum().sum() / X.size * 100:.2f}%)"
)
print("\n[컬럼 결측률 분포]")
print(miss.describe().round(4).to_string())
print("\n[구간별 컬럼 수]")
bins = pd.cut(
    miss,
    [-0.001, 0, 0.05, 0.2, 0.5, 0.9, 1.0],
    labels=["0%", "0~5%", "5~20%", "20~50%", "50~90%", "90~100%"],
)
print(bins.value_counts().sort_index().to_string())
print(f"\n상수 컬럼(고유값 ≤ 1) : {int((nuniq <= 1).sum())}개")
print(
    f"준상수 컬럼(최빈값 99% 이상): "
    f"{int((X.apply(lambda c: c.value_counts(normalize=True, dropna=True).max() if c.notna().any() else 1) > 0.99).sum())}개"
)
print(
    f"\n행 대비 열 비율 : {X.shape[1]}열 / {X.shape[0]}행 = {X.shape[1] / X.shape[0]:.2f}"
)
print(f"불량 1건당 센서 : {X.shape[1] / y.sum():.1f}개")
print("""
★★ 불량 104건인데 센서가 590개입니다.
   이 상황에서는 '우연히 불량과 잘 맞는 센서'가 반드시 나옵니다.
   고차원 저표본에서는 과적합이 기본값입니다. A7-4에서 실측합니다.""")

# ----------------------------------------------------------------------
rule("A7-3. 정제 — 무엇을 버릴지 규칙부터 정합니다")
steps = []


def log(name, cols):
    steps.append({"단계": name, "남은 컬럼": len(cols)})
    return cols


cols = log("0. 원본", list(X.columns))
cols = log("1. 상수 컬럼 제거", [c for c in cols if nuniq[c] > 1])
cols = log("2. 결측 50% 초과 제거", [c for c in cols if miss[c] <= 0.5])
# 중복 컬럼(값이 완전히 같은 센서) 제거
sub = X[cols]
dup = sub.T.duplicated()
cols = log("3. 중복 컬럼 제거", [c for c in cols if not dup[c]])
print(pd.DataFrame(steps).to_string(index=False))
print(f"\n590개 → {len(cols)}개 ({len(cols) / 590 * 100:.1f}% 생존)")

Xc = X[cols].copy()
print(f"남은 결측: {int(Xc.isna().sum().sum()):,}개 → 중앙값 대체 예정")
print("""
★ 왜 평균이 아니라 중앙값인가: 공정 센서는 이상치가 흔하고 분포가 치우쳐 있습니다.
★ 왜 대체를 '파이프라인 안'에서 하는가: 전체 중앙값으로 미리 채우면
  test 정보가 train에 새어 들어갑니다(A3-8과 같은 문제).""")

# ----------------------------------------------------------------------
rule("A7-4. ★★ 다중검정 함정 — 590번 검정하면 우연이 나옵니다")
pvals = {}
for c in cols:
    a = Xc.loc[y == 0, c].dropna()
    b = Xc.loc[y == 1, c].dropna()
    if len(a) > 5 and len(b) > 5 and (a.std() > 0 or b.std() > 0):
        pvals[c] = stats.ttest_ind(a, b, equal_var=False).pvalue
pv = pd.Series(pvals).dropna().sort_values()
n_test = len(pv)
sig05 = int((pv < 0.05).sum())
print(f"검정한 센서 수      : {n_test}개")
print(f"p < 0.05 인 센서    : {sig05}개")
print(f"우연히 기대되는 개수 : {n_test * 0.05:.1f}개  (검정 수 × 0.05)")
print(f"→ 실제 신호는 많아야 {sig05 - n_test * 0.05:.0f}개 정도입니다.")

# Bonferroni & BH(FDR)
bonf = 0.05 / n_test
n_bonf = int((pv < bonf).sum())
m = n_test
ranked = pv.reset_index(drop=True)
bh_thresh = (np.arange(1, m + 1) / m) * 0.05
below = ranked.values < bh_thresh
n_bh = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
print(f"\nBonferroni 보정 (α={bonf:.2e}) 통과 : {n_bonf}개")
print(f"BH(FDR 5%) 통과                    : {n_bh}개")
print("\n[p값 상위 10개 센서]")
top = pv.head(10)
tt = pd.DataFrame(
    {
        "p값": top.round(6),
        "정상 평균": [round(Xc.loc[y == 0, c].mean(), 3) for c in top.index],
        "불량 평균": [round(Xc.loc[y == 1, c].mean(), 3) for c in top.index],
    }
)
print(tt.to_string())
print(f"""
★★ p<0.05가 {sig05}개나 나왔지만, {n_test}번 검정했으니 {n_test * 0.05:.0f}개는 그냥 우연입니다.
   Bonferroni를 통과한 건 {n_bonf}개뿐입니다.
   "유의한 센서를 {sig05}개 찾았다"고 쓰면 그건 통계를 모르는 겁니다.

   ★ 실무 권장: Bonferroni는 너무 보수적이라 진짜 신호도 죽입니다.
     BH(FDR) 절차가 현실적인 절충입니다 → 여기서는 {n_bh}개.""")

# ----------------------------------------------------------------------
rule("A7-5. 모델링 — 시간순 split (라벨에 타임스탬프가 있습니다)")
ordr = lab["ts"].argsort().values
Xo = Xc.iloc[ordr].reset_index(drop=True)
yo = y.iloc[ordr].reset_index(drop=True)
ts_o = lab["ts"].iloc[ordr].reset_index(drop=True)
cut = int(len(Xo) * 0.75)
print(
    f"train {cut}건 ({ts_o.iloc[0].date()} ~ {ts_o.iloc[cut - 1].date()}) "
    f"불량 {int(yo[:cut].sum())}건 ({yo[:cut].mean() * 100:.2f}%)"
)
print(
    f"test  {len(Xo) - cut}건 ({ts_o.iloc[cut].date()} ~ {ts_o.iloc[-1].date()}) "
    f"불량 {int(yo[cut:].sum())}건 ({yo[cut:].mean() * 100:.2f}%)"
)


def make(m):
    return Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", m)]
    )


cands = {
    "무조건 정상": DummyClassifier(strategy="most_frequent"),
    "로지스틱": make(LogisticRegression(max_iter=2000, random_state=SEED)),
    "로지스틱+가중": make(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    ),
    "랜덤포레스트": Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "m",
                RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
            ),
        ]
    ),
    "랜덤포레스트+가중": Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "m",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}
rows = []
for name, mdl in cands.items():
    mdl.fit(Xo[:cut], yo[:cut])
    if hasattr(mdl, "predict_proba"):
        p = mdl.predict_proba(Xo[cut:])[:, 1]
    else:
        p = mdl.predict(Xo[cut:])
    pred = (p >= 0.5).astype(int)
    yt = yo[cut:]
    rows.append(
        {
            "모델": name,
            "정확도": round((pred == yt).mean(), 4),
            "재현율": round(recall_score(yt, pred, zero_division=0), 4),
            "F1": round(f1_score(yt, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(yt, p), 4) if len(set(p)) > 1 else 0.5,
            "PR-AUC": round(average_precision_score(yt, p), 4),
        }
    )
res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))
base = yo[cut:].mean()
print(f"\nPR-AUC 기준선(= test 불량률) = {base:.4f}")
best = res.loc[res["PR-AUC"].idxmax()]
print(f"""
★★★ 결론부터: 잘 안 나옵니다.
   최고 PR-AUC {best["PR-AUC"]:.4f} (모델: {best["모델"]}), 기준선 {base:.4f}.
   기준선 대비 {best["PR-AUC"] / base:.2f}배입니다. AI4I(0.90 / 기준선 0.034 = 26배)와
   비교하면 사실상 '거의 못 맞힌다'에 가깝습니다.

   정확도는 여전히 93% 넘게 나옵니다. 여기서도 정확도는 아무 의미가 없습니다.""")

# ----------------------------------------------------------------------
rule("A7-6. 왜 안 나오나 — 원인을 짚어 봅니다")
print("""가설 1) 신호 자체가 약하다
가설 2) 표본이 부족하다 (불량 104건)
가설 3) 시간에 따라 공정이 변한다 (분포 이동)""")

# 가설 1 검증: 효과크기
eff = []
for c in pv.head(20).index:
    a = Xc.loc[y == 0, c].dropna()
    b = Xc.loc[y == 1, c].dropna()
    sp = np.sqrt(
        ((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(len(a) + len(b) - 2, 1)
    )
    eff.append(abs(a.mean() - b.mean()) / sp if sp > 0 else 0)
print(f"\n[가설1] p값 상위 20개 센서의 Cohen's d 중앙값 = {np.median(eff):.3f}")
print("  (0.2=작음 0.5=중간 0.8=큼)  → 가장 강한 센서조차 효과가 작습니다.")

# 가설 3 검증: 전반부/후반부 불량률과 분포
half = len(yo) // 2
print(
    f"\n[가설3] 전반부 불량률 {yo[:half].mean() * 100:.2f}% / "
    f"후반부 불량률 {yo[half:].mean() * 100:.2f}%"
)
shift = []
for c in pv.head(50).index:
    a = Xo.loc[:half, c].dropna()
    b = Xo.loc[half:, c].dropna()
    if len(a) > 10 and len(b) > 10:
        shift.append(stats.ks_2samp(a, b).pvalue)
shift = np.array(shift)
print(
    f"  상위 50개 센서 중 전/후반 분포가 유의하게 다른 것: "
    f"{int((shift < 0.05).sum())}개 ({(shift < 0.05).mean() * 100:.0f}%)"
)
print(
    "  → 공정 자체가 시간에 따라 변합니다. 앞 시기로 배운 걸 뒤 시기에 쓰기 어렵습니다."
)

# 랜덤 split과 비교 (누수 확인)
Xtr, Xte, ytr, yte = train_test_split(
    Xc, y, test_size=0.25, random_state=SEED, stratify=y
)
mdl = Pipeline(
    [
        ("imp", SimpleImputer(strategy="median")),
        ("m", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)),
    ]
)
mdl.fit(Xtr, ytr)
pr = mdl.predict_proba(Xte)[:, 1]
print(
    f"\n[참고] 랜덤 split PR-AUC {average_precision_score(yte, pr):.4f} "
    f"vs 시간순 split {best['PR-AUC']:.4f}"
)
print("  SECOM은 로트 간격이 수십 분~수 시간이라 자기상관이 약합니다.")
print("  그래서 A3의 시뮬레이터만큼 극적인 차이는 안 납니다.")

# ----------------------------------------------------------------------
rule("A7-7. 그래도 건질 것 — 순위 지표로 보면")
mdl_best = cands["랜덤포레스트"]
p_best = mdl_best.predict_proba(Xo[cut:])[:, 1]
yt = yo[cut:].values
order = np.argsort(-p_best)
rows = []
for k in [10, 20, 30, 50, 100]:
    hit = int(yt[order][:k].sum())
    rows.append(
        {
            "상위 K": k,
            "잡은 불량": hit,
            "Precision@K": round(hit / k, 3),
            "Recall@K": round(hit / max(yt.sum(), 1), 3),
            "무작위 기대": round(k * yt.mean(), 1),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
lift50 = yt[order][:50].sum() / max(50 * yt.mean(), 1e-9)
lift100 = yt[order][:100].sum() / max(100 * yt.mean(), 1e-9)
print(f"""
★ 기대를 걸었지만 여기도 아닙니다. 그대로 씁니다.
  상위 50건 검사 → 불량 {int(yt[order][:50].sum())}건. 무작위 기대 {50 * yt.mean():.1f}건.
  향상도(lift) {lift50:.2f}배 — 사실상 무작위와 같습니다.
  상위 100건도 {int(yt[order][:100].sum())}건 vs 무작위 {100 * yt.mean():.1f}건 (lift {lift100:.2f}배).

★★ 순위 지표로 바꿔 봐도 살아나지 않습니다.
   "지표를 바꿔서 좋아 보이게 만드는 것"과 "안 되는 걸 인정하는 것"은 다릅니다.
   이 데이터는 후자입니다.""")

# ----------------------------------------------------------------------
rule("A7-8. ★ 이 절에서 남길 것 (보고서에 이렇게 씁니다)")
print(f"""
  "SECOM 데이터로 웨이퍼 불량을 예측했으나 실용 수준의 성능을 얻지 못했습니다
   (시간순 split PR-AUC {best["PR-AUC"]:.3f}, 기준선 {base:.3f}).

   원인 분석 결과
   (1) 개별 센서의 효과크기가 작습니다(상위 20개 Cohen's d 중앙값 {np.median(eff):.2f}).
   (2) 불량 표본이 104건뿐인데 센서는 590개로, 고차원 저표본입니다.
   (3) 상위 센서의 {(shift < 0.05).mean() * 100:.0f}%에서 전/후반 분포가 유의하게 달라
       공정 자체가 시간에 따라 변합니다.

   순위 기반 검사 전략도 확인했으나 향상도가 {lift50:.2f}배로 무작위와 다르지 않아
   대안이 되지 못했습니다.

   따라서 이 데이터만으로는 불량 예측 모델을 운영에 넣을 수 없다고 판단합니다.
   필요한 것은 더 좋은 모델이 아니라 (a) 불량 표본 확대,
   (b) 센서 태그의 의미 정보(어느 공정 단계인지), (c) 공정 변경 이력입니다."

★★ 이렇게 쓴 포트폴리오가, 억지로 SMOTE 돌려서 "F1 0.9 달성"이라 쓴 것보다
   훨씬 좋은 평가를 받습니다. 면접관은 성능이 아니라 판단력을 봅니다.

★★★ 학생이 가장 많이 하는 실수: 성능이 안 나오면 데이터를 바꿉니다.
   그러지 마세요. '안 되는 이유를 규명한 분석'은 그 자체로 완결된 결과물입니다.
   실무에서 하는 일의 절반이 이겁니다.""")

# ----------------------------------------------------------------------
rule("A7-9. 그림")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].hist(miss, bins=50, color="#c53030")
axes[0].set_yscale("log")
axes[0].set_xlabel("컬럼 결측률")
axes[0].set_ylabel("컬럼 수 (log)")
axes[0].set_title("SECOM 결측 구조")

axes[1].hist(pv.values, bins=40, color="#2b6cb0", edgecolor="white")
axes[1].axhline(
    n_test / 40 * 1.0, color="gray", ls="--", label="균등분포(=신호 없음) 기대선"
)
axes[1].axvline(0.05, color="crimson", ls="--", label="p=0.05")
axes[1].set_xlabel("p값")
axes[1].set_ylabel("센서 수")
axes[1].set_title(f"590개 센서의 p값 분포 (n={n_test})")
axes[1].legend(fontsize=8)

ks_ = np.arange(1, len(yt) + 1)
axes[2].plot(
    ks_,
    np.cumsum(yt[order]) / max(yt.sum(), 1),
    color="#2b6cb0",
    lw=1.8,
    label="모델 순위",
)
axes[2].plot(
    ks_, ks_ * yt.mean() / max(yt.sum(), 1), color="gray", ls="--", label="무작위"
)
axes[2].set_xlabel("검사 건수 K")
axes[2].set_ylabel("잡은 불량 비율")
axes[2].set_title("SECOM — 순위 성능 (미미합니다)")
axes[2].legend(fontsize=8)
fig.tight_layout()
save(fig, "secom")

res.to_csv(ROOT / "data" / "secom_result.csv", index=False)
print("저장: data/secom_result.csv")


# ======================================================================
# 실행 결과 정리
# ======================================================================
# SECOM은 불량 104건에 센서 590개로 고차원 저표본 문제를 확인함
# p<0.05 센서는 79개였지만 다중검정 보정 후 Bonferroni 6개, BH(FDR) 22개로 감소
# 시간순 split의 최고 PR-AUC는 0.0833으로 기준선 0.0612보다 소폭 높은 수준에 그침
# 상위 센서의 효과 크기가 작고 92%에서 전후반 분포 변화가 나타나 성능 저하 원인을 확인함
# 상위 50건의 lift도 0.98배로 순위 기반 검사 역시 무작위보다 개선되지 않음
# 따라서 현재 데이터만으로 운영 모델을 만들기 어렵고 불량 표본과 센서 의미 정보, 공정 변경 이력이 더 필요하다고 판단됨
