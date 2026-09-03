"""
설비 센서 시뮬레이터 (물리 기반)
================================
CNC 밀링 설비 3대를 1분 단위로 시뮬레이션합니다.

두 가지 결과를 동시에 만듭니다.
  - truth    : 오염 없는 참값 (정답지)
  - observed : 현장에서 실제로 받는 더러운 데이터

참값을 갖고 있으므로 "내 전처리가 얼마나 원래 값을 되찾았는지"를
수치로 검증할 수 있습니다. 현실에서는 불가능한 사치인데,
학습용으로는 이것만큼 좋은 게 없습니다.

고장 규칙은 UCI AI4I 2020 데이터셋의 정의를 따랐습니다.
(Matzka, S. 2020) 그래야 Part 2의 실데이터와 바로 이어집니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 설비 스펙
# ----------------------------------------------------------------------
MACHINES = {
    # machine_id : (품질등급, 마모한계 계수, 공구교체주기(분))
    "CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
    "CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
    "CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}

# 관측 오염 강도 (기본값 = "현장급")
POLLUTION = {
    "dropout_rate": 0.015,  # 통신 끊김으로 통째로 사라지는 구간 발생 확률
    "dropout_len": (3, 40),  # 끊김 길이(분)
    "nan_rate": 0.008,  # 개별 센서값만 NaN
    "spike_rate": 0.004,  # 센서 튐(전기 노이즈)
    "dup_rate": 0.006,  # 같은 레코드 중복 전송
    "ts_jitter_rate": 0.05,  # 타임스탬프 흔들림
    "unit_mix_rate": 0.10,  # 단위 혼재(K 대신 섭씨로 오는 구간)
    "drift_per_day": 0.35,  # 온도 센서 드리프트 (K/day)
}


# ----------------------------------------------------------------------
# 1) 물리 기반 참값 생성
# ----------------------------------------------------------------------
def _simulate_one(
    machine_id: str, n_minutes: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    spec = MACHINES[machine_id]
    ts = pd.date_range(start, periods=n_minutes, freq="min")

    # --- 공정 부하: 근무 시간대에 높고 야간에 낮음 (일주기) ---
    hour = ts.hour + ts.minute / 60.0
    duty = 0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)  # 0.1 ~ 1.0
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    # --- 공기 온도: 계절/일교차 + 랜덤워크 ---
    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = air + np.cumsum(rng.normal(0, 0.02, n_minutes))  # 완만한 표류
    air = air + rng.normal(0, 0.15, n_minutes)

    # --- 공구 마모: 누적되다가 교체하면 0으로 ---
    tool_life = spec["tool_life"]
    wear_rate = 1.0 + 0.6 * duty  # 부하 클수록 빨리 닳음
    wear = np.zeros(n_minutes)
    acc = rng.uniform(0, 60)  # 시작 시점 마모도는 랜덤
    limit = tool_life * rng.uniform(0.90, 1.15)
    for i in range(n_minutes):
        acc += wear_rate[i]
        if acc > limit:  # 계획 교체 (정비반 재량으로 조금씩 다름)
            acc = 0.0
            limit = tool_life * rng.uniform(0.90, 1.15)
        wear[i] = acc

    # --- 회전수: 부하에 반비례(무거운 절삭일수록 저속) ---
    rpm = 2860 - 1500 * duty + rng.normal(0, 45, n_minutes)
    rpm = np.clip(rpm, 1150, 2900)

    # --- 토크: 부하에 비례, 마모되면 저항 증가 ---
    torque = 10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes)
    torque = np.clip(torque, 3.0, 80.0)

    # --- 냉각(HVAC) 이상: 가끔 공장 공조가 죽어 실내가 더워짐 ---
    hvac_fail = np.zeros(n_minutes, dtype=bool)
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s : s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail  # 실내 온도 상승

    # --- 공정 온도: 공기온도 + 절삭열. 쿨런트가 process 쪽은 어느 정도 잡아줌 ---
    power_w = torque * rpm * 2 * np.pi / 60.0  # [W]
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear
    proc = proc - 6.0 * hvac_fail  # 온도차(방열 여력)가 줄어듦
    proc = proc + rng.normal(0, 0.12, n_minutes)

    # --- 진동: 마모·회전수에 비례. 마모 후반에 급격히 커짐 ---
    vib = (
        0.8
        + 0.0009 * rpm
        + 0.9 * (wear / tool_life) ** 3
        + rng.normal(0, 0.06, n_minutes)
    )
    vib = np.clip(vib, 0.1, None)

    # --- 전류: 전력/전압(380V, 역률 0.85, 3상) ---
    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

    # --- 습도: 온도와 약한 음의 관계 ---
    humid = 55 - 1.8 * (air - 298) + rng.normal(0, 2.5, n_minutes)
    humid = np.clip(humid, 15, 95)

    df = pd.DataFrame(
        {
            "ts": ts,
            "machine_id": machine_id,
            "type": spec["type"],
            "air_temp_k": air,
            "process_temp_k": proc,
            "rot_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": wear,
            "vibration_mms": vib,
            "current_a": current,
            "humidity_pct": humid,
        }
    )

    # ------------------------------------------------------------------
    # 고장 라벨 (AI4I 2020 정의 그대로)
    # ------------------------------------------------------------------
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power_w < 3500) | (power_w > 9000)
    osf = (wear * torque) > spec["osf_limit"]
    rnf = rng.random(n_minutes) < 0.0002  # 원인 불명 랜덤 고장

    df["twf"] = twf.astype(int)
    df["hdf"] = hdf.astype(int)
    df["pwf"] = pwf.astype(int)
    df["osf"] = osf.astype(int)
    df["rnf"] = rnf.astype(int)
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df


def simulate_truth(
    n_minutes: int = 1440, start: str | pd.Timestamp = "2024-01-01", seed: int = 42
) -> pd.DataFrame:
    """오염 없는 참값을 생성합니다."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    parts = [_simulate_one(m, n_minutes, start, rng) for m in MACHINES]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2) 현장급 오염 주입
# ----------------------------------------------------------------------
SENSOR_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
]


def pollute(
    truth: pd.DataFrame,
    seed: int = 7,
    cfg: dict | None = None,
    return_masks: bool = False,
):
    """참값에 현장에서 실제로 생기는 오염을 주입합니다.

    주입 순서가 곧 현실의 발생 순서입니다.
      드리프트(설비) → 단위혼재(수집기 설정) → 튐(전기노이즈)
      → 결측(센서) → 끊김(통신) → 중복/타임스탬프(전송)

    return_masks=True면 "어디에 무엇을 주입했는지" 정답지를 함께 돌려줍니다.
    전처리 성능을 정밀도·재현율로 채점하기 위한 것입니다.
    """
    c = dict(POLLUTION)
    if cfg:
        c.update(cfg)
    rng = np.random.default_rng(seed)
    df = truth.copy()
    masks = pd.DataFrame(index=df.index)

    # 관측 데이터에는 참값 라벨 중 machine_failure만 남깁니다.
    # (현장에서도 세부 고장코드는 정비 후에야 붙습니다)
    df = df.drop(columns=["twf", "hdf", "pwf", "osf", "rnf", "power_w"])

    t0 = df["ts"].min()
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0

    # --- (a) 센서 드리프트: CNC-02 온도 센서만 서서히 밀림 ---
    m2 = df["machine_id"] == "CNC-02"
    df.loc[m2, "process_temp_k"] += c["drift_per_day"] * days[m2]

    # --- (b) 단위 혼재: 특정 구간에서 온도가 섭씨로 들어옴 ---
    n = len(df)
    unit_block = np.zeros(n, dtype=bool)
    n_blocks = max(1, int(n * c["unit_mix_rate"] / 200))
    for _ in range(n_blocks):
        s = rng.integers(0, n - 200)
        unit_block[s : s + 200] = True
    df.loc[unit_block, "air_temp_k"] -= 273.15
    df.loc[unit_block, "process_temp_k"] -= 273.15
    masks["unit_temp"] = unit_block
    # 진동 단위도 일부는 m/s^2 로 (×9.81)
    vib_block = rng.random(n) < 0.04
    df.loc[vib_block, "vibration_mms"] *= 9.81
    masks["unit_vib"] = vib_block

    # --- (c) 센서 튐: 값이 순간적으로 10~50배 또는 0 ---
    for col in SENSOR_COLS:
        hit = rng.random(n) < c["spike_rate"]
        mode = rng.random(n)
        df.loc[hit & (mode < 0.5), col] = df.loc[hit & (mode < 0.5), col] * rng.uniform(
            8, 40
        )
        df.loc[hit & (mode >= 0.5), col] = 0.0
        masks[f"spike_{col}"] = hit

    # --- (d) 개별 결측 ---
    for col in SENSOR_COLS:
        hit = rng.random(n) < c["nan_rate"]
        df.loc[hit, col] = np.nan
        masks[f"nan_{col}"] = hit

    # --- (e) 통신 끊김: 행 자체가 사라짐 ---
    drop_mask = np.zeros(n, dtype=bool)
    n_drop = int(n * c["dropout_rate"] / 10)
    for _ in range(max(1, n_drop)):
        s = rng.integers(0, n)
        ln = rng.integers(*c["dropout_len"]) * 3  # 설비 3대 × 분
        drop_mask[s : s + ln] = True
    masks["dropped"] = drop_mask
    keep = ~drop_mask
    df = df[keep].copy()
    kept_masks = masks[keep].copy()

    # --- (f) 중복 전송 ---
    n2 = len(df)
    dup_idx = rng.random(n2) < c["dup_rate"]
    dups = df[dup_idx].copy()
    kept_masks["is_dup"] = False
    dup_masks = kept_masks[dup_idx].copy()
    dup_masks["is_dup"] = True
    df = pd.concat([df, dups], ignore_index=True)
    kept_masks = pd.concat([kept_masks, dup_masks], ignore_index=True)

    # --- (g) 타임스탬프 흔들림 + 순서 뒤섞임 ---
    n3 = len(df)
    jitter = np.where(
        rng.random(n3) < c["ts_jitter_rate"], rng.integers(-90, 90, n3), 0
    )
    df["ts"] = df["ts"] + pd.to_timedelta(jitter, unit="s")
    kept_masks["ts_jittered"] = jitter != 0
    order = rng.permutation(n3)
    df = df.iloc[order].reset_index(drop=True)
    kept_masks = kept_masks.iloc[order].reset_index(drop=True)

    # --- (h) 실제 수집기가 붙이는 메타 컬럼 ---
    df["collected_at"] = pd.Timestamp("2024-01-01")
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")  # 문자열로 들어옴(현실)
    if return_masks:
        return df, kept_masks
    return df


# ----------------------------------------------------------------------
# 3) 실시간 수집용: "지금부터 n분 치"
# ----------------------------------------------------------------------
def sample_window(
    n_minutes: int = 60, end: pd.Timestamp | None = None, seed: int | None = None
) -> pd.DataFrame:
    """수집기가 호출하는 함수. 최근 n분 구간의 관측 데이터를 돌려줍니다."""
    end = pd.Timestamp.utcnow().floor("min") if end is None else pd.Timestamp(end)
    start = end - pd.Timedelta(minutes=n_minutes)
    # 시드를 날짜에서 뽑으면 같은 날 다시 돌려도 같은 값이 나옵니다(재현성)
    if seed is None:
        seed = int(start.strftime("%Y%m%d%H"))
    truth = simulate_truth(n_minutes=n_minutes, start=start, seed=seed)
    obs = pollute(truth, seed=seed + 1)
    obs["collected_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return obs


if __name__ == "__main__":
    t = simulate_truth(n_minutes=1440, start="2024-01-01", seed=42)
    o = pollute(t, seed=7)
    print("truth   :", t.shape)
    print("observed:", o.shape)
    print(o.head(3).to_string())


# -------------------------------------------------
# 깨끗한 참값 데이터인 truth를 만들고
# 거기에 결측/중복/오염을 넣은 observed를 만든 다음 일부를 보여준 결과

# truth   : (4320, 18)
# 1440분 * 3대 = 4320행 -> 3대 설비의 하루치 1분 단위 원본 참값 데이터

# observed: (3983, 13)
# dropout등의 처리를 했기 때문에 truth보다 행 수가 줄어든 걸 확인
# 실제로 pullute() 함수에 누락, 중복, 시간 흔들림 같은 오염 코드를 작성함

#                     ts machine_id type  air_temp_k  process_temp_k  rot_speed_rpm  torque_nm  tool_wear_min  vibration_mms  current_a  humidity_pct  machine_failure collected_at
# 0  2024-01-01 13:17:00     CNC-02    M  298.625851      313.422285    1406.052695  49.563036     214.189992       2.694331  12.776496     54.273886                0   2024-01-01
# 1  2024-01-01 17:31:00     CNC-03    H  299.465280      314.813583    1863.785516  43.060944     228.093245       3.187122  15.337023     53.284947                0   2024-01-01
# 2  2024-01-01 04:57:00     CNC-02    M  296.731156      310.754775    2237.713224  30.457300     136.175066       3.042451  12.643848     57.330195                0   2024-01-01
# 현재 출력된 표는 observed의 일부임에 주의!
