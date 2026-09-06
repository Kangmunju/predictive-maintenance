# python -m experiments.clean_validation

import numpy as np
import pandas as pd

from src.simulator import simulate_truth, pollute, SENSOR_COLS
from src.clean import (
    run_pipeline,
    PHYS_RANGE,
    detect_and_fix_temp_unit,
    detect_vibration_unit,
    hampel_flag,
)


# 검증용 데이터 생성
truth = simulate_truth(n_minutes=14 * 1440, start="2024-01-01", seed=42)
# 14일치 오염 없는 참값 생성
obs, mask = pollute(truth, seed=7, return_masks=True)  # 참값에 오염 주입
# 탐지 규칙 검증 과정에서 return_masks=True를 추가함
clean, log, rep = run_pipeline(obs)  # 오염된 데이터를 정제 파이프라인에 적용
#           단계    행수    증감
#      0. 원본 수신 55471     0
#      1. 타입 강제 55471     0
#   2. 타임스탬프 스냅 55471     0
#      3. 중복 제거 53327 -2144
#  4a. 온도 단위 통일 53327     0
#  4b. 진동 단위 통일 53327     0
# 5. 물리범위 → NaN 53327     0
#   6. 스파이크 플래그 53327     0
#   7. 짧은 결측 보간 53327     0
#    8. 드리프트 보정 53327     0
#    9. 시간축 재색인 60480  7153


# 참값과 정제 결과 연결
sens = SENSOR_COLS
t = truth.copy()
t["ts"] = t["ts"].dt.round("min")
m = clean.merge(
    t[["machine_id", "ts"] + sens],
    on=["machine_id", "ts"],
    how="inner",
    suffixes=("_c", "_t"),
)
# 참값의 시간 기준을 정제 데이터와 동일하게 맞춤
# 설비와 측정 시각을 기준으로 병합하여 정제값과 참값을 비교할 수 있도록 구성

print("대조 가능 행 :", f"{len(m):,} / 참값 {len(truth):,}")
# 대조 가능 행 : 60,480 / 참값 60,480


# 센서별 정제 결과 검증
comp_rows = []

for col in sens:
    clean_col = f"{col}_c"  # 정제된 센서값 컬럼
    truth_col = f"{col}_t"  # 참값 컬럼
    valid = m[clean_col].notna() & m[truth_col].notna()
    # 정제값과 참값이 모두 존재하는 행만 비교
    err = (m.loc[valid, clean_col] - m.loc[valid, truth_col]).abs()
    # 정제값과 참값의 절대 오차
    value_rate = valid.mean() * 100  # 실제 비교 가능한 값이 남아 있는 비율
    mae = err.mean()  # 평균 절대 오차
    p95_err = err.quantile(0.95)  # 오차의 95% 지점
    max_err = err.max()
    # 참값의 표준편차
    truth_std = m.loc[valid, truth_col].std()
    mae_std = mae / truth_std if truth_std != 0 else np.nan
    # 데이터의 자연 변동에 비해 오차가 얼마나 큰지 비교
    comp_rows.append(
        {
            "센서": col,
            "값보유율(%)": round(value_rate, 2),
            "MAE": round(mae, 4),
            "p95_err": round(p95_err, 4),
            "max_err": round(max_err, 3),
            "참값std": round(truth_std, 3),
            "MAE/std": round(mae_std, 4),
        }
    )
comp = pd.DataFrame(comp_rows)
print()
print(comp.to_string(index=False))
#             센서  값보유율(%)    MAE  p95_err  max_err   참값std  MAE/std
#     air_temp_k    88.17 0.0597   0.0000   33.351   2.281   0.0262
# process_temp_k    88.17 0.0226   0.0484    2.167   2.075   0.0109
#  rot_speed_rpm    88.17 2.8330   0.0000  367.759 481.832   0.0059
#      torque_nm    88.17 0.0905   0.0000   14.982  13.024   0.0069
#  tool_wear_min    88.17 0.3968   0.0000  352.969  68.086   0.0058
#  vibration_mms    88.17 0.0059   0.0000    1.982   0.526   0.0112
#      current_a    88.17 0.0311   0.0000    5.637   2.246   0.0139
#   humidity_pct    88.17 0.1867   0.0000   63.390   4.790   0.0390

# 정제 데이터와 참값을 비교하여 센서별 복원 성능을 검증함
# 값보유율과 오차 지표를 통해 정제 결과가 참값에 얼마나 가깝게 복원되었는지 확인함


# 정제 전후 비교

before = obs.copy()
before["ts"] = pd.to_datetime(before["ts"], errors="coerce").dt.round("min")
before = before.merge(
    t[["machine_id", "ts"] + sens],
    on=["machine_id", "ts"],
    how="inner",
    suffixes=("_o", "_t"),
)
compare_rows = []
for col in sens:
    before_col = f"{col}_o"
    truth_col = f"{col}_t"
    valid_before = before[before_col].notna() & before[truth_col].notna()
    # 두 값이 모두 존재하는 행만 비교
    before_mae = (
        (before.loc[valid_before, before_col] - before.loc[valid_before, truth_col])
        .abs()
        .mean()
    )
    after_mae = comp.loc[comp["센서"] == col, "MAE"].iloc[0]
    # 앞에서 계산한 정제 후 MAE 가져옴
    improvement = before_mae / after_mae if after_mae != 0 else np.nan
    # 정제 전 오차가 정제 후보다 몇 배 큰지 계산
    compare_rows.append(
        {
            "센서": col,
            "정제전MAE": round(before_mae, 3),
            "정제후MAE": round(after_mae, 4),
            "개선배수": round(improvement, 1),
        }
    )
compare = pd.DataFrame(compare_rows)
print()
print("정제 전후 비교")
print(compare.to_string(index=False))
# 정제 전후 비교
#             센서  정제전MAE  정제후MAE   개선배수
#     air_temp_k  36.919  0.0597  618.4
# process_temp_k  35.523  0.0226 1571.8
#  rot_speed_rpm 147.847  2.8330   52.2
#      torque_nm   2.447  0.0905   27.0
#  tool_wear_min   2.475  0.3968    6.2
#  vibration_mms   1.127  0.0059  191.0
#      current_a   0.965  0.0311   31.0
#   humidity_pct   1.749  0.1867    9.4

# 정제 전후의 MAE를 비교하여 정제 파이프라인 적용
# 센서별 오차가 실제로 얼마나 감소했는지 확인


# 탐지 규칙 검증
def score(pred, gt, name):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {
        "규칙": name,
        "실제": int(gt.sum()),
        "탐지": int(pred.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "정밀도": round(tp / (tp + fp), 3),  # 오염이라고 예측한 것 중 실제 오염
        "재현율": round(tp / (tp + fn), 3),  # 실제로 심어둔 오염 중 찾아낸 오염
    }


# pred : 작성한 규칙의 탐지 결과
# mask : 실제로 심어둔 오염 위치

# 온도 단위 오류 탐지 검증
air_pred = pd.to_numeric(obs["air_temp_k"], errors="coerce") < 200
process_pred = pd.to_numeric(obs["process_temp_k"], errors="coerce") < 200
air_score = score(air_pred, mask["unit_temp"], "air_temp_k < 200")
process_score = score(process_pred, mask["unit_temp"], "process_temp_k < 200")
unit_score = pd.DataFrame([air_score, process_score])
print()
print("온도 단위 오류 탐지")
print(unit_score.to_string(index=False))
# 온도 단위 오류 탐지
#                   규칙   실제   탐지   TP  FP  FN   정밀도   재현율
#     air_temp_k < 200 5910 5943 5841 102  69 0.983 0.988
# process_temp_k < 200 5910 5937 5850  87  60 0.985 0.990

# 온도가 200보다 작으면 섭씨 단위가 잘못 섞인 것으로 판단
# 위 규칙이 실제 주입된 단위 오류를 상당히 정확히 탐지하고 있다는 것을 확인


# 진동 단위 오류 탐지 검증
vib = pd.to_numeric(obs["vibration_mms"], errors="coerce")
vib_med = vib.groupby(obs["machine_id"]).transform("median")
vib_pred = vib > vib_med * 4
vib_score = score(vib_pred, mask["unit_vib"], "vibration > median × 4")
print()
print("진동 단위 오류 탐지")
print(pd.DataFrame([vib_score]).to_string(index=False))
# 진동 단위 오류 탐지
#                     규칙   실제   탐지   TP  FP  FN   정밀도   재현율
# vibration > median × 4 2178 2272 2162 110  16 0.952 0.993

# 설비별 진동 중앙값의 4배를 초과하는 값을 단위 오류로 판단하고 실제 오염 위치와 비교
# 실제 진동 단위 오류 2,178개 중 2,162개를 탐지하여 재현율 99.3% 확인
# 탐지한 값 중 실제 단위 오류의 비율은 95.2%로 높은 탐지 성능을 확인


# 온도 단위 오류 임계값 비교
temp_threshold_rows = []
for threshold in [150, 180, 200, 250, 273]:
    temp_pred = pd.to_numeric(obs["air_temp_k"], errors="coerce") < threshold
    result = score(
        temp_pred,
        mask["unit_temp"],
        f"air_temp_k < {threshold}",
    )
    temp_threshold_rows.append(result)
temp_threshold = pd.DataFrame(temp_threshold_rows)
print()
print("온도 임계값 비교")
print(temp_threshold.to_string(index=False))
# 온도 임계값 비교
#               규칙   실제   탐지   TP  FP  FN   정밀도   재현율
# air_temp_k < 150 5910 5943 5841 102  69 0.983 0.988
# air_temp_k < 180 5910 5943 5841 102  69 0.983 0.988
# air_temp_k < 200 5910 5943 5841 102  69 0.983 0.988
# air_temp_k < 250 5910 5943 5841 102  69 0.983 0.988
# air_temp_k < 273 5910 5943 5841 102  69 0.983 0.988

# 온도 단위 탐지 기준을 150~273으로 변경해가며 성능을 비교
# 현재 데이터에서는 정상 켈빈값과 섭씨 혼입값이 충분히 분리되어 있음
# 임계값을 변경해도 정밀도와 재현율이 동일하게 나타나는 것을 확인


# 진동 단위 오류 임계값 비교
vib_threshold_rows = []
for ratio in [1.2, 1.5, 2, 4, 6]:
    vib_pred = vib > vib_med * ratio
    result = score(
        vib_pred,
        mask["unit_vib"],
        f"vibration > median × {ratio}",
    )
    vib_threshold_rows.append(result)
vib_threshold = pd.DataFrame(vib_threshold_rows)
print()
print("진동 임계값 비교")
print(vib_threshold.to_string(index=False))
# 진동 임계값 비교
#                       규칙   실제   탐지   TP   FP  FN   정밀도   재현율
# vibration > median × 1.2 2178 8295 2162 6133  16 0.261 0.993
# vibration > median × 1.5 2178 2342 2162  180  16 0.923 0.993
#   vibration > median × 2 2178 2272 2162  110  16 0.952 0.993
#   vibration > median × 4 2178 2272 2162  110  16 0.952 0.993
#   vibration > median × 6 2178 2272 2162  110  16 0.952 0.993

# 진동 단위 오류 기준을 중앙값의 1.2~6배로 변경해가며 탐지 성능을 비교
# 1.2배 기준에서는 정상값까지 많이 탐지되어 정밀도가 크게 낮아짐
# 1.5배 이상에서는 오탐이 감소
# 2배 이상에서는 정밀도와 재현율이 동일
# 진동값은 실제 설비 이상에 의해 높아질 수 있으므로 임계값을너무 낮게 설정하면 오탐이 증가할 수도 있음을 확인


# 실제 베어링 이상 상황에서 진동 단위 오류 규칙의 한계 확인
o4 = obs.copy()
fault_idx = o4.index[(o4["machine_id"] == "CNC-03") & o4["vibration_mms"].notna()][:400]
o4.loc[fault_idx, "vibration_mms"] *= 5.0
# CNC-03의 정상 진동값 400개를 5배 증가시켜 실제 고장 상황을 가정
med4 = o4.groupby("machine_id")["vibration_mms"].transform("median")
rule_hit = o4["vibration_mms"] > med4 * 4
# 설비별 중앙값의 4배를 넘으면 단위 오류라고 판단
fault_detected = rule_hit.loc[fault_idx].sum()
print()
print("베어링 이상 오인 확인")
print("실제 고장으로 만든 행 :", len(fault_idx))
print("단위 오류로 잘못 탐지 :", fault_detected)
print("오인 비율 :", round(fault_detected / len(fault_idx) * 100, 1), "%")

# 실제 베어링 이상 상황을 가정
# 기존 진동 단위 오류 규칙을 적용한 결과 00개 중 336개(84%)를단위 오류로 잘못 탐지
# 통계적 기준만으로 단위를 판정하면 실제 설비 이상 신호까지 단위 오류로 오인할 수 있음을 확인


# 스파이크 탐지 검증 - 물리 범위 검사
range_data = obs.copy()
range_data, _ = detect_and_fix_temp_unit(range_data)
range_data, _ = detect_vibration_unit(range_data)
# 물리 범위 검사 전에 온도와 진동의 단위를 먼저 정상화
range_score_rows = []
for col in sens:
    lo, hi = PHYS_RANGE[col]
    range_pred = range_data[col].notna() & ~pd.to_numeric(
        range_data[col], errors="coerce"
    ).between(lo, hi)
    result = score(
        range_pred,
        mask[f"spike_{col}"],
        f"{col} 범위검사",
    )
    range_score_rows.append(result)
range_score = pd.DataFrame(range_score_rows)
print()
print("스파이크 탐지 - 범위 검사")
print(range_score.to_string(index=False))
# 스파이크 탐지 - 범위 검사
#                  규칙  실제  탐지  TP  FP  FN  정밀도   재현율
#     air_temp_k 범위검사 220 105 105   0 115  1.0 0.477
# process_temp_k 범위검사 208 207 207   0   1  1.0 0.995
#  rot_speed_rpm 범위검사 236 235 235   0   1  1.0 0.996
#      torque_nm 범위검사 219 218 218   0   1  1.0 0.995
#  tool_wear_min 범위검사 215  92  92   0 123  1.0 0.428
#  vibration_mms 범위검사 198  88  88   0 110  1.0 0.444
#      current_a 범위검사 218 217 217   0   1  1.0 0.995
#   humidity_pct 범위검사 225 109 109   0 116  1.0 0.484

# 물리 범위 검사는 오탐 없이 스파이크를 탐지
# 센서 특성에 따라 일부 스파이크를 놓쳐 재현율 차이가 발생함을 확인


# 스파이크 탐지 검증 - 물리 범위 검사 + Hampel
hampel_data = obs.copy()
hampel_data["ts"] = pd.to_datetime(
    hampel_data["ts"],
    errors="coerce",
)
hampel_data, _ = detect_and_fix_temp_unit(hampel_data)
hampel_data, _ = detect_vibration_unit(hampel_data)
# Hampel 적용 전에 센서 단위를 먼저 정상화
hampel_data = hampel_data.sort_values(["machine_id", "ts"])
# 설비별 시간 순서로 정렬
# 원래 index는 유지하여 mask와 같은 행끼리 비교할 수 있도록 함
hampel_score_rows = []
for col in sens:
    lo, hi = PHYS_RANGE[col]
    range_pred = hampel_data[col].notna() & ~hampel_data[col].between(lo, hi)
    # 물리 범위를 벗어난 값 탐지
    hampel_pred = hampel_data.groupby("machine_id")[col].transform(
        lambda s: hampel_flag(s, window=11, n_sigma=5.0)
    )
    # 설비별 시간 흐름에서 주변 값과 크게 다른 급격한 변화 탐지
    total_pred = range_pred | hampel_pred
    # 범위 검사 또는 Hampel 중 하나라도 탐지하면 스파이크로 판단
    gt = mask.loc[
        hampel_data.index,
        f"spike_{col}",
    ]
    # 정렬 후에도 원래 index를 이용해 실제 스파이크 위치와 연결
    result = score(
        total_pred,
        gt,
        f"{col} 범위+Hampel",
    )
    hampel_score_rows.append(result)
hampel_score = pd.DataFrame(hampel_score_rows)
print()
print("스파이크 탐지 - 범위 + Hampel")
print(hampel_score.to_string(index=False))
# 스파이크 탐지 - 범위 + Hampel
#                       규칙  실제  탐지  TP  FP  FN   정밀도   재현율
#     air_temp_k 범위+Hampel 220 635 220 415   0 0.346 1.000
# process_temp_k 범위+Hampel 208 576 207 369   1 0.359 0.995
#  rot_speed_rpm 범위+Hampel 236 553 235 318   1 0.425 0.996
#      torque_nm 범위+Hampel 219 565 218 347   1 0.386 0.995
#  tool_wear_min 범위+Hampel 215 286 200  86  15 0.699 0.930
#  vibration_mms 범위+Hampel 198 609 195 414   3 0.320 0.985
#      current_a 범위+Hampel 218 575 217 358   1 0.377 0.995
#   humidity_pct 범위+Hampel 225 546 224 322   1 0.410 0.996

# Hampel을 함께 사용하면 놓치는 스파이크는 줄어들지만 정상값까지 탐지하는 경우가 증가함
# 범위 검사는 값 처리에 사용하고 Humpel은 이상 여부를 확인하기 위한 플래그로 활용하였음


# 행 수 회계 - 최종 검산
log_df = log.frame()
truth_rows = len(truth)
# 오염을 넣기 전 참값 전체 행 수
observed_rows = len(obs)
# 통신 끊김과 중복 등이 반영된 관측 데이터 행 수
dedup_rows = log_df.loc[
    log_df["단계"] == "3. 중복 제거",
    "행수",
].iloc[0]
# 타임스탬프 정리 후 중복을 제거한 행 수
final_rows = len(clean)
# 시간축 재색인까지 완료한 최종 행 수
value_rows = dedup_rows
# 실제 센서값이 존재하던 행 수
value_rate = value_rows / final_rows * 100
# 최종 시간축 중 실제 값이 존재하는 비율
missing_rate = 100 - value_rate
# 실제 측정값이 존재하지 않는 시간 비율
print()
print("행 수 회계")
print("참값 :", f"{truth_rows:,}")
print("관측 :", f"{observed_rows:,}")
print("중복 제거 후 :", f"{dedup_rows:,}")
print("최종 재색인 후 :", f"{final_rows:,}")
print("실제 값 존재 :", f"{value_rows:,}")
print("값 보유율 :", round(value_rate, 2), "%")
print("결측 비율 :", round(missing_rate, 2), "%")
# 행 수 회계
# 참값 : 60,480
# 관측 : 55,471
# 중복 제거 후 : 53,327
# 최종 재색인 후 : 60,480
# 실제 값 존재 : 53,327
# 값 보유율 : 88.17 %
# 결측 비율 : 11.83 %

# 통신 끊김, 중복으로 인해 관측 행 수가 변경되었고 중복 제거 후 실제 값이 있는 행 53,327개
# 재색인은 빠진 시간대를 빈 행으로 추가하여 전체 시간축을 복원하며 실제 측정값을 새로 만드는 것은 아님


# --------------------------------------------------------------
# 참값과 정제 결과를 비교해 정제 후 센서 오차가 크게 감소하는 것을 확인
# MAE뿐 아니라 p95와 max 오차를 함께 확인해 일부 큰 오차가 남아 았는 지도 확인하였음

# 온도 단위 오류 : 물리적으로 불가능한 값이라는 기준을 사용할 수 있어 안정적으로 탐지 가능
# 진동 단위 오류 : 통계 기준을 사용하는 경우 실제 설비 이상까지 단위 오류로 오인할 가능성 있음
# 가능하면 태그 단위표 같은 메타 데이터를 사용하고 보정한 행은 따로 기록할 필요가 있음을 확인

# 물리 범위 검사 : 오탐이 거의 없으나 센서 특성에 따라 일부 스파이크를 놓칠 가능성 존재
# Hampel을 함께 사용하면 놓치는 스파이크는 감소하지만 정상적인 급격한 변화까지 탐지해버림
# 따라서 범위 검사는 실제 값 처리에 사용하고 Hampel은 이상 여부를 확인하는 플래그로 사용할 것
# 또한 단위 통일 이후 적용해야 정상 데이터를 이상값으로 오인하지 않는다는 것을 확인함
# 탐지 규칙을 검증할 때에도 실제 정제 파이프라인의 처리 순서와 데이터 정렬을 고려할 것

# 빠진 시간대를 빈 행으로 다시 만들어 전체 시간축을 맞추는 재색인 과정을 추가
# 사라진 실제 측정값을 복구하지는 않았음
# 최종 행 수뿐 아니라 각 단계에서 행 수가 변한 이유를 설명할 수 있도록 함
