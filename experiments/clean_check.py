# python -m experiments.clean_check

import pandas as pd
import numpy as np

from src.simulator import simulate_truth, pollute
from src.clean import run_pipeline

truth = simulate_truth(n_minutes=14 * 1440, start="2024-01-01", seed=42)

obs = pollute(truth, seed=7)

clean, log, rep = run_pipeline(obs)

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


print("[단위 보정 건수]", rep["temp_unit"], "| 진동:", rep["vib_unit"])
# [단위 보정 건수] {'air_temp_k': 5697, 'process_temp_k': 5692} | 진동: 2179
print(pd.Series(rep["range"]).to_string())  # 물리범위 위반인 경우 NaN
# air_temp_k         98
# process_temp_k    202
# rot_speed_rpm     228
# torque_nm         208
# tool_wear_min      88
# vibration_mms      85
# current_a         209
# humidity_pct      105
print(pd.Series(rep["filled"]).to_string())  # 짧은 결측 보간
# air_temp_k        531
# process_temp_k    658
# rot_speed_rpm     663
# torque_nm         643
# tool_wear_min     526
# vibration_mms     521
# current_a         649
# humidity_pct      517
print(pd.Series(rep["drift_slopes"]).round(4).to_string())
# CNC-01   -0.0119
# CNC-02    0.3463
# CNC-03    0.0000

# 정제 파이프라인 9단계의 실행 결과와 단계별 행 수 변화를 확인함
# 단위 보정, 물리 범위, 결측 보간, 드리프트 보정 결과가 교안 예시와 동일한 것을 확인함
