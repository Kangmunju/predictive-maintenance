# python -m experiments.pollution_check


from src.simulator import *
# src.simulator에서 작성한 함수와 변수들을 오염도체크에서 사용함

truth = simulate_truth(n_minutes=1440 * 14, start="2024-01-01", seed=42)
# simulator.py에서 정의한 truth 사용
# 24시간 * 60분 * 14일

obs, masks = pollute(truth, seed=7, return_masks=True)
print("참값 행수 :", f"{len(truth):,}")  # 참값 행수 : 60,480
print(
    "관측 행수 :", f"{len(obs):,}", f"({len(obs) - len(truth):+,})"
)  # 관측 행수 : 55,471 (-5,009)

# 5,009행이 사라진 것을 확인

print(obs.head(3).to_string())
#                    ts machine_id type  air_temp_k  process_temp_k  rot_speed_rpm  torque_nm  tool_wear_min  vibration_mms  current_a  humidity_pct  machine_failure collected_at
# 0  2024-01-01 19:27:00     CNC-01    L   26.277953       40.388514    2203.556139  31.066702     120.798056       2.856060  12.769520     52.529610                0   2024-01-01
# 1  2024-01-07 17:24:00     CNC-01    L  302.688464      316.292073    1959.352887  33.437922       6.877461       2.540470  12.460443     44.471779                0   2024-01-01
# 2  2024-01-04 21:50:00     CNC-03    H   26.386493       38.781660    2570.807775  20.710829      18.742839       3.078863  10.051499     55.304309                0   2024-01-01
# -> 정렬안됨, ts문자열, 단위혼재

print((obs[SENSOR_COLS].isna().mean() * 100).round(2).to_string())
# 센서별 결측률 확인
# air_temp_k        0.81
# process_temp_k    0.87
# rot_speed_rpm     0.82
# torque_nm         0.81
# tool_wear_min     0.83
# vibration_mms     0.82
# current_a         0.82
# humidity_pct      0.77
# isna()는 아예 존재하지 않는 행을 결측치로 세지 못함
# dropout으로 빠진 행 + 중복으로 추가된 행 + 그 외 -> 최종 행 수 차이 -5,009행
# 위 사유들로 인해 결측률이 0.08%로 낮게 나온 것으로 추측함

print(obs["air_temp_k"].describe().round(2).to_string())
# air_temp_k 값의 분포 확인
# count    55022.00
# mean       276.79
# std        204.42
# min          0.00
# 25%        296.49
# 50%        298.51
# 75%        300.20
# max       5054.47
# 대부분의 값이 296 ~ 300K 근처에 몰려있는 것을 확인
# 그런데 min=0.0 max=5054.47d을 보아 이상값이 껴있다는 것 또한 확인
print("200 K 미만 비율 : %.2f%%" % ((obs["air_temp_k"] < 200).mean() * 100))
# 현재 관측된 air_temp_k 값 중 200K보다 미만인 값의 비율 : 10.71%
