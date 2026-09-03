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
# 대부분의 값이 296 ~ 300K 근처에 몰려있는 것 확인
# min=0.0 max=5054.47d을 보아 이상값이 껴있다는 것 확인
# 평균이 중앙값보다 훨씬 적은 것 확인
print("200 K 미만 비율 : %.2f%%" % ((obs["air_temp_k"] < 200).mean() * 100))
# 현재 관측된 air_temp_k 값 중 200K보다 미만인 값의 비율 : 10.71%

inj = pd.DataFrame({"건수": masks.sum(), "비율(%)": (masks.mean() * 100).round(3)})
print(inj.to_string)
#                          건수   비율(%)
# unit_temp             5910  10.654
# unit_vib              2178   3.926
# spike_air_temp_k       220   0.397
# spike_process_temp_k   208   0.375
# spike_rot_speed_rpm    236   0.425
# spike_torque_nm        219   0.395
# spike_tool_wear_min    215   0.388
# spike_vibration_mms    198   0.357
# spike_current_a        218   0.393
# spike_humidity_pct     225   0.406
# nan_air_temp_k         449   0.809
# nan_process_temp_k     484   0.873
# nan_rot_speed_rpm      454   0.818
# nan_torque_nm          452   0.815
# nan_tool_wear_min      460   0.829
# nan_vibration_mms      455   0.820
# nan_current_a          453   0.817
# nan_humidity_pct       429   0.773
# dropped                  0   0.000
# is_dup                 367   0.662
# ts_jittered           2791   5.031>


# df = truth[truth["machine_id"] == "CNC-02"]
# df["process_temp_k"] += 0.35
# 교안에서는 SettingWithCopyWarning 발생 예시이나 현재 환경에서는 경고 발생하지 않음

print("\n[행 순서 섞임 확인]")
print("시간순 정렬 여부 :", obs["ts"].is_monotonic_increasing)
# 시간순 정렬 여부 : False

print("\n[데이터와 마스크 행수 확인]")
print("관측 데이터 행수 :", len(obs))
print("마스크 행수 :", len(masks))
print("행수 일치 여부 :", len(obs) == len(masks))
# 관측 데이터 행수 : 55471
# 마스크 행수 : 55471
# 행수 일치 여부 : True

print("\n[행 삭제 및 중복 검산]")
dup_count = int(masks["is_dup"].sum())  # 중복으로 추가된 행 개수
drop_count = len(truth) + dup_count - len(obs)
# 관측 행수 = 참값 행수 - 삭제 행수 + 중복 행수
# 따라서 삭제 행수 = 참값 행수 + 중복 행수 - 관측 행수
check_count = len(truth) - drop_count + dup_count
# 삭제와 중복을 적용했을 때 최종 행수가 맞는지 다시 계산

print("\n[재검산]")
print("참값 행수 :", len(truth))
print("끊김으로 삭제 :", drop_count)
print("중복으로 추가 :", dup_count)
print("검산 :", check_count)
print("실제 관측 행수 :", len(obs))
# [재검]
# 참값 행수 : 60480
# 끊김으로 삭제 : 5376
# 중복으로 추가 : 367
# 검산 : 55471
# 실제 관측 행수 : 55471
