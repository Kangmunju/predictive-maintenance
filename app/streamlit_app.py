"""
설비 예지보전 대시보드
======================
Streamlit Community Cloud 무료 배포용.

배포 시 주의
  - 저장소 루트 기준 경로를 씁니다(상대경로 금지 → __file__ 기준으로 계산)
  - 무거운 학습은 여기서 하지 않습니다. models/*.csv 를 읽기만 합니다
    (Community Cloud는 메모리 1GB 제한이 있어 학습을 돌리면 죽습니다)
  - 한글 폰트가 없으므로 matplotlib 대신 Streamlit 내장 차트를 씁니다
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
MODELS = ROOT / "models"

st.set_page_config(page_title="설비 예지보전 대시보드", layout="wide")


# ----------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
    files = sorted(HIST.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    num = [
        "air_temp_k",
        "process_temp_k",
        "rot_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "vibration_mms",
        "current_a",
        "humidity_pct",
    ]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts"]).drop_duplicates(subset=["machine_id", "ts"])


@st.cache_data(ttl=600)
def load_metrics() -> dict:
    p = MODELS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=600)
def load_preds() -> pd.DataFrame:
    p = MODELS / "test_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, parse_dates=["ts"])
    return d


@st.cache_data(ttl=600)
def load_importance() -> pd.DataFrame:
    p = MODELS / "feature_importance.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, index_col=0)


raw = load_history()
met = load_metrics()
preds = load_preds()

st.title("설비 예지보전 대시보드")
st.caption(
    "CNC 밀링 3대 · 1분 단위 센서 · 30분 내 고장 예지 "
    "| 데이터는 물리 기반 시뮬레이터에서 생성됩니다(합성 데이터)"
)

if raw.empty:
    st.error(
        "data/history/*.csv 가 없습니다. `python src/collector.py --minutes 1440` 를 먼저 실행하세요."
    )
    st.stop()

# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["운영 현황", "데이터 품질", "모델 성능", "방법·한계"])

with tab1:
    c = st.columns(4)
    c[0].metric("수집 행수", f"{len(raw):,}")
    c[1].metric("설비 수", raw["machine_id"].nunique())
    c[2].metric("수집 기간(일)", (raw["ts"].max() - raw["ts"].min()).days + 1)
    fr = raw["machine_failure"].mean() * 100 if "machine_failure" in raw else np.nan
    c[3].metric("고장 발생률", f"{fr:.2f}%")

    st.markdown("#### 설비별 센서 추이")
    mid = st.selectbox("설비", sorted(raw["machine_id"].unique()))
    sensor = st.selectbox(
        "센서",
        [
            "torque_nm",
            "vibration_mms",
            "process_temp_k",
            "rot_speed_rpm",
            "current_a",
            "tool_wear_min",
        ],
    )
    sub = raw[raw["machine_id"] == mid].sort_values("ts")
    lo, hi = st.select_slider(
        "기간",
        options=list(sub["ts"].dt.date.unique()),
        value=(sub["ts"].dt.date.min(), sub["ts"].dt.date.max()),
    )
    m = sub[(sub["ts"].dt.date >= lo) & (sub["ts"].dt.date <= hi)]
    st.line_chart(m.set_index("ts")[[sensor]], height=260)

    st.markdown("#### 설비별 일간 고장 건수")
    if "machine_failure" in raw:
        daily = (
            raw.assign(d=raw["ts"].dt.date)
            .groupby(["d", "machine_id"])["machine_failure"]
            .sum()
            .unstack()
        )
        st.bar_chart(daily, height=260)

with tab2:
    st.markdown("#### 결측률 (원본 수신 기준)")
    num = [
        "air_temp_k",
        "process_temp_k",
        "rot_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "vibration_mms",
        "current_a",
        "humidity_pct",
    ]
    num = [c for c in num if c in raw.columns]
    miss = (raw[num].isna().mean() * 100).round(2).rename("결측률(%)")
    st.bar_chart(miss, height=240)

    st.markdown("#### 수집 공백 — 통신이 끊긴 구간")
    g = raw.sort_values(["machine_id", "ts"]).copy()
    g["gap_min"] = g.groupby("machine_id")["ts"].diff().dt.total_seconds() / 60
    gaps = g[g["gap_min"] > 2][["machine_id", "ts", "gap_min"]]
    cc = st.columns(3)
    cc[0].metric("2분 넘는 공백", f"{len(gaps):,}회")
    cc[1].metric("최장 공백", f"{gaps['gap_min'].max():.0f}분" if len(gaps) else "-")
    cc[2].metric(
        "총 결손 시간", f"{gaps['gap_min'].sum() / 60:.1f}시간" if len(gaps) else "-"
    )
    st.dataframe(gaps.sort_values("gap_min", ascending=False).head(15), height=260)

    st.markdown("#### 단위 혼재 흔적")
    if "air_temp_k" in raw:
        n_c = int((raw["air_temp_k"] < 200).sum())
        st.write(
            f"공기온도가 200 K 미만인 행: **{n_c:,}건** "
            f"({n_c / len(raw) * 100:.2f}%) → 섭씨로 들어온 구간입니다."
        )
        st.bar_chart(
            pd.cut(raw["air_temp_k"].dropna(), bins=40)
            .value_counts()
            .sort_index()
            .rename("건수")
            .reset_index(drop=True),
            height=200,
        )

with tab3:
    if not met:
        st.warning(
            "models/metrics.json 이 없습니다. `python src/train.py` 를 실행하세요."
        )
    else:
        st.markdown("#### 성능 (★ 시간순 split 기준)")
        c = st.columns(4)
        c[0].metric("PR-AUC", met.get("pr_auc"))
        c[1].metric("ROC-AUC", met.get("roc_auc"))
        c[2].metric("기준선(양성률)", met.get("positive_rate_test"))
        c[3].metric("적용 임계값", met.get("threshold"))
        c = st.columns(3)
        c[0].metric("정밀도", met.get("precision"))
        c[1].metric("재현율", met.get("recall"))
        c[2].metric("F1", met.get("f1"))

        st.info(
            f"임계값은 0.5가 아니라 **{met.get('threshold')}** 입니다. "
            f"고장 놓침(800만원)이 헛점검(30만원)보다 26배 비싸다는 가정에서 "
            f"총비용이 최소가 되는 지점입니다. "
            f"0.5를 쓰면 비용이 {met.get('cost_at_0.5', 0) / 1e8:.1f}억, "
            f"이 임계값이면 {met.get('cost_at_threshold', 0) / 1e8:.1f}억입니다."
        )

        if not preds.empty:
            st.markdown("#### 예측 확률 분포 — 실제 고장 여부별")
            b = np.linspace(0, 1, 26)
            h0 = np.histogram(preds.loc[preds["y"] == 0, "prob"], bins=b)[0]
            h1 = np.histogram(preds.loc[preds["y"] == 1, "prob"], bins=b)[0]
            st.bar_chart(
                pd.DataFrame(
                    {"정상": h0, "30분내 고장": h1}, index=np.round(b[:-1], 2)
                ),
                height=260,
            )

            st.markdown("#### 상위 K건 점검 시 성능")
            yt = preds.sort_values("prob", ascending=False)["y"].values
            ks = [10, 25, 50, 100, 200, 500]
            rows = [
                {
                    "상위 K": k,
                    "잡은 고장": int(yt[:k].sum()),
                    "Precision@K": round(yt[:k].mean(), 3),
                    "무작위 기대": round(k * yt.mean(), 1),
                }
                for k in ks
                if k <= len(yt)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        imp = load_importance()
        if not imp.empty:
            st.markdown("#### 변수 중요도 상위 15")
            st.bar_chart(imp.head(15), height=300)

with tab4:
    st.markdown(f"""
### 무엇을 한 프로젝트인가
CNC 밀링 설비 3대의 1분 단위 센서 데이터를 수집·정제하고,
**"앞으로 {met.get("horizon_min", 30)}분 안에 고장이 발생할지"** 를 예측합니다.

### 데이터
- **합성 데이터입니다.** 물리 기반 시뮬레이터(`src/simulator.py`)가 생성합니다.
  고장 규칙은 UCI AI4I 2020 데이터셋의 정의를 따랐습니다.
- 시뮬레이터는 의도적으로 현장급 오염을 주입합니다:
  통신 끊김, 센서 튐, 타임스탬프 중복·흔들림, 단위 혼재(K↔℃), 센서 드리프트.
- GitHub Actions가 매일 자동 수집해 `data/history/`에 쌓습니다.

### 이 프로젝트에서 신경 쓴 것
1. **정확도를 쓰지 않습니다.** 불균형 데이터에서 정확도는 모델을 구분하지 못합니다.
   PR-AUC와 기준선(양성률) 대비로 봅니다.
2. **시간순 split을 씁니다.** 랜덤 split을 쓰면 PR-AUC가 0.98까지 올라가지만
   그건 옆자리 답을 본 것입니다(실측 대조는 저장소의 분석 리포트 참조).
3. **임계값을 0.5로 쓰지 않습니다.** 고장 놓침과 헛점검의 비용이 다르므로
   총비용 최소점에서 정합니다.
4. **이상치를 함부로 지우지 않습니다.** 설비 고장은 이상치의 모습으로 나타납니다.

### 한계 (반드시 읽어 주세요)
- **합성 데이터입니다.** 실제 설비 데이터가 아니므로 성능 수치를 현장에 그대로
  적용할 수 없습니다. 파이프라인 설계와 검증 방법이 이 프로젝트의 결과물입니다.
- 비용 가정(고장 800만원 / 헛점검 30만원)은 **가정**입니다. 실제 값은 정비팀에서
  받아야 합니다. 비용비가 바뀌면 임계값도 바뀝니다.
- 시뮬레이터의 고장 규칙이 결정론적이라, 실제 설비보다 예측이 쉽습니다.
  실데이터(AI4I·SECOM) 분석을 함께 수행한 이유입니다.
- 3대·2주 데이터라 계절성·장기 열화를 반영하지 못합니다.
""")
    if met:
        st.json(met)

st.caption("소스: https://github.com/Kangmunju/predictive-maintenance")
