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
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(ttl=600)
def load_preds() -> pd.DataFrame:
    p = MODELS / "test_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["ts"])


@st.cache_data(ttl=600)
def load_importance() -> pd.DataFrame:
    p = MODELS / "feature_importance.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, index_col=0)


def fmt_num(value, digits=4, suffix="") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}{suffix}"


def fmt_count(value, suffix="건") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}{suffix}"


def fmt_cost(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) / 1e8:,.2f}억 원"


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
    c[2].metric("데이터 존재 일수", f"{raw['ts'].dt.normalize().nunique():,}일")
    st.caption(
        f"데이터 범위: {raw['ts'].min():%Y-%m-%d} ~ "
        f"{raw['ts'].max():%Y-%m-%d} | "
        "2024년 초기 시뮬레이션과 2026년 자동 수집 데이터를 포함하며, "
        "중간의 미수집 기간은 데이터 존재 일수에 포함하지 않습니다."
    )
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
        c[0].metric("PR-AUC (AP)", fmt_num(met.get("pr_auc")))
        c[1].metric("ROC-AUC", fmt_num(met.get("roc_auc")))
        c[2].metric("기준선(양성률)", fmt_num(met.get("positive_rate_test")))
        c[3].metric("적용 임계값", fmt_num(met.get("threshold"), 2))
        c = st.columns(3)
        c[0].metric("정밀도", fmt_num(met.get("precision")))
        c[1].metric("재현율", fmt_num(met.get("recall")))
        c[2].metric("F1", fmt_num(met.get("f1")))

        st.caption(
            "위 정밀도·재현율·F1은 개별 행(관측 시점) 기준입니다. 아래 고장 에피소드 평가와 집계 단위가 다릅니다."
        )

        st.markdown("#### 예측 방법별 AP 비교")
        ap_values = {
            "양성률 기준선": met.get("positive_rate_test"),
            "단순 이력 규칙": met.get("pr_auc_naive"),
            "RandomForest": met.get("pr_auc"),
        }
        ap_values = {k: float(v) for k, v in ap_values.items() if v is not None}
        if len(ap_values) == 3:
            st.bar_chart(pd.Series(ap_values, name="AP"), height=280)
            st.caption(
                "단순 이력 규칙: 해당 설비의 직전 30개 관측치에서 실제 고장 발생 비율을 점수로 사용합니다. "
                "현재 데이터셋의 과거 정답 라벨을 활용한 비교 기준으로, 현장 배포 가능한 독립 예측 모델을 의미하지 않습니다."
            )
            st.write(
                f"RandomForest는 단순 이력 규칙보다 AP가 "
                f"**{fmt_num(met.get('ap_improvement_over_naive'))}p** 높습니다 "
                "(0~1 척도에서의 절대 차이)."
            )
        else:
            st.info(
                "AP 비교에 필요한 지표가 일부 없습니다. 모델 재학습 결과를 확인하세요."
            )

        st.markdown("#### 행 단위 비용 비교")
        row_costs = {
            f"임계값 {fmt_num(met.get('threshold'), 2)}": met.get("cost_at_threshold"),
            "임계값 0.50": met.get("cost_at_0.5"),
        }
        row_costs = {k: float(v) / 1e8 for k, v in row_costs.items() if v is not None}
        if row_costs:
            st.bar_chart(pd.Series(row_costs, name="비용(억 원)"), height=240)
        st.info(
            f"고장 놓침 800만원, 불필요한 점검 30만원을 가정한 **행 단위 비용**입니다. "
            f"임계값 {fmt_num(met.get('threshold'), 2)}에서 {fmt_cost(met.get('cost_at_threshold'))}, "
            f"0.50에서 {fmt_cost(met.get('cost_at_0.5'))}입니다. "
            "이 임계값은 테스트 구간의 행 단위 비용을 최소화하도록 선택했으므로, "
            "독립 검증 데이터에서 확정한 운영 임계값이 아닙니다."
        )

        episode = met.get("episode_evaluation") or {}
        episode_05 = met.get("episode_evaluation_at_0.5") or {}
        if episode:
            st.markdown("#### 고장 에피소드 탐지 평가")
            c = st.columns(4)
            c[0].metric("고장 에피소드", fmt_count(episode.get("failure_episodes")))
            c[1].metric("탐지 성공", fmt_count(episode.get("detected_episodes")))
            c[2].metric("미탐지", fmt_count(episode.get("missed_episodes")))
            c[3].metric(
                "에피소드 탐지율",
                fmt_num(
                    episode.get("episode_recall") * 100
                    if episode.get("episode_recall") is not None
                    else None,
                    2,
                    "%",
                ),
            )
            if (
                episode.get("detected_episodes") is not None
                and episode.get("missed_episodes") is not None
            ):
                st.bar_chart(
                    pd.Series(
                        {
                            "탐지 성공": episode["detected_episodes"],
                            "미탐지": episode["missed_episodes"],
                        },
                        name="에피소드 수",
                    ),
                    height=220,
                )

            st.markdown("#### 알람 중복 제거 및 오경보")
            c = st.columns(3)
            c[0].metric("독립 알람 그룹", fmt_count(episode.get("alarm_groups")))
            c[1].metric("오경보", fmt_count(episode.get("false_alarms")))
            c[2].metric(
                "일평균 오경보",
                fmt_num(episode.get("false_alarms_per_day"), 2, "건/일"),
            )
            st.caption(
                "고장 발생 시점 간격 30분 이내는 하나의 고장 에피소드로 묶습니다. "
                "알람은 같은 설비에서 10분 이내 간격으로 이어지면 하나의 그룹으로 묶고, "
                "고장 전 30분 구간에 활성화된 알람 그룹과 고장 에피소드를 일대일 매칭합니다. "
                "평가 구간 양 끝의 30분 경계는 평가 대상에서 제외하는 규칙을 적용합니다."
            )

            st.markdown("#### 에피소드 기준 비용 비교")
            ep_costs = {}
            for label, item in [
                (f"임계값 {fmt_num(episode.get('threshold'), 2)}", episode),
                ("임계값 0.50", episode_05),
            ]:
                if item.get("episode_cost") is not None:
                    ep_costs[label] = float(item["episode_cost"]) / 1e8
            if ep_costs:
                st.bar_chart(pd.Series(ep_costs, name="비용(억 원)"), height=240)
            c = st.columns(2)
            c[0].metric(
                "적용 임계값의 에피소드 비용", fmt_cost(episode.get("episode_cost"))
            )
            c[1].metric(
                "임계값 0.50의 에피소드 비용", fmt_cost(episode_05.get("episode_cost"))
            )
            st.caption(
                "에피소드 비용 = 미탐지 에피소드 × 800만원 + 오경보 알람 그룹 × 30만원. "
                "행 단위 비용과는 집계 단위가 달라 금액을 직접 비교할 수 없습니다. "
                "두 임계값만 비교한 결과이며 에피소드 비용의 전역 최소값을 탐색한 결과는 아닙니다."
            )
        else:
            st.warning(
                "에피소드 평가 결과가 없습니다. 최신 src/train.py 실행 결과를 확인하세요."
            )

        if not preds.empty and {"y", "prob"}.issubset(preds.columns):
            st.markdown("#### 예측 확률 분포 — 실제 고장 여부별")
            b = np.linspace(0, 1, 26)
            h0 = np.histogram(preds.loc[preds["y"] == 0, "prob"].dropna(), bins=b)[0]
            h1 = np.histogram(preds.loc[preds["y"] == 1, "prob"].dropna(), bins=b)[0]
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
  고장 규칙은 UCI AI4I 2020 데이터셋의 정의를 참고했습니다.
- 시뮬레이터는 의도적으로 현장급 오염을 주입합니다:
  통신 끊김, 센서 튐, 타임스탬프 중복·흔들림, 단위 혼재(K↔℃), 센서 드리프트.
- GitHub Actions가 매일 자동 수집해 `data/history/`에 쌓습니다.

### 이 프로젝트에서 신경 쓴 것
1. **정확도를 쓰지 않습니다.** 불균형 데이터에서 정확도는 모델을 구분하지 못합니다.
   PR-AUC와 기준선(양성률) 대비로 봅니다.
2. **시간순 split을 씁니다.** 랜덤 split을 쓰면 PR-AUC가 0.98까지 올라가지만
   그건 옆자리 답을 본 것입니다(실측 대조는 저장소의 분석 리포트 참조).
3. **임계값을 0.5로 고정하지 않습니다.** 고장 놓침과 헛점검의 비용이 다르므로
   비용을 고려합니다. 다만 현재 임계값은 테스트 구간에서 선택한 값입니다.
4. **이상치를 함부로 지우지 않습니다.** 설비 고장은 이상치의 모습으로 나타납니다.
5. **행 단위 성능과 고장 에피소드 성능을 구분합니다.** 연속 경보를 하나로 묶고
   실제 고장 사건을 탐지했는지 별도로 평가합니다.

### 한계 (반드시 읽어 주세요)
- **합성 데이터입니다.** 실제 설비 데이터가 아니므로 성능 수치를 현장에 그대로
  적용할 수 없습니다. 파이프라인 설계와 검증 방법이 이 프로젝트의 결과물입니다.
- 비용 가정(고장 800만원 / 헛점검 30만원)은 **가정**입니다. 실제 값은 정비팀에서
  받아야 합니다. 비용비가 바뀌면 임계값도 바뀝니다.
- 시뮬레이터의 고장 규칙이 결정론적이라, 실제 설비보다 예측이 쉽습니다.
  실데이터(AI4I·SECOM) 분석을 함께 수행한 이유입니다.
- 기존 설명의 3대·2주 데이터는 초기 실험 기준이며, 자동 수집으로 현재 데이터 기간은
  늘어날 수 있습니다. 실제 수집 기간은 운영 현황 탭의 값이 기준입니다.
- **테스트 구간에서 임계값을 선택**했으므로 비용 평가가 낙관적일 수 있습니다.
  향후 시간순 학습·검증·테스트 분리로 개선해야 합니다.
- 시간순 분할만으로 모든 누수가 제거되는 것은 아닙니다. 경계 시점의 미래 라벨,
  전처리 통계량 및 동일 시각 설비 간 분할 문제는 별도 점검이 필요합니다.
- 에피소드 평가는 30분 사건 묶기, 10분 알람 묶기 및 경계 제외 정책에 따라 달라집니다.
""")
    if met:
        st.json(met)

st.caption("소스: https://github.com/Kangmunju/predictive-maintenance")


# =============================================================================
# [구현 핵심]
# =============================================================================

# 1. 대시보드 경량화
#    - Streamlit Cloud의 메모리 제한을 고려하여 모델 재학습 없이 저장된 CSV/JSON만 로드
#    - @st.cache_data(ttl=600)을 적용하여 반복적인 파일 읽기 최소화
#
# 2. 모델 성능 비교
#    - 양성률 기준선, 과거 30개 관측치 기반 단순 규칙, RandomForest의 AP 비교
#    - 행 단위 정밀도·재현율·F1과 에피소드 단위 탐지율을 구분하여 시각화
#
# 3. 알람 및 비용 평가
#    - 독립 알람 그룹, 오경보 건수, 일평균 오경보를 운영 지표로 표시
#    - 행 단위 비용과 에피소드 단위 비용을 분리하여 잘못된 직접 비교 방지
#    - 테스트 구간에서 임계값을 선택한 평가상의 한계 명시
#
# 4. 데이터 수집 기간 계산 개선
#    - 문제: 최초~최종 날짜 차이로 계산하여 미수집 기간까지 포함
#    - 원인: 2024년 초기 시뮬레이션과 2026년 자동 수집 데이터가 혼재
#    - 해결: ts.dt.normalize().nunique()로 실제 데이터 존재 날짜만 집계
#    - 결과: 기존 1,014일 → 실제 데이터 존재 일수 49일로 수정
#    - 최초·최종 날짜를 별도 표시하여 전체 데이터 범위와 수집 일수 구분
#
# 5. 호환성 및 검증
#    - metrics.json에 에피소드 평가 결과가 없는 경우 경고를 표시하고 기존 기능 유지
#    - 문법 검사: python -m py_compile app/streamlit_app.py
#    - 실행 검증: python -m streamlit run app/streamlit_app.py
#    - 확인 결과: 대시보드 정상 실행 및 데이터 존재 일수 49일 표시 확인
