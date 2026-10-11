# python src/download_data.py

"""
공개 데이터셋 다운로드
======================
    python src/download_data.py

data/raw/ 는 .gitignore에 들어 있습니다(용량·저작권).
이 스크립트가 있으면 누구나 같은 상태를 재현할 수 있습니다.
★ 포트폴리오에서 "재현 가능한가"는 생각보다 크게 봅니다.

다운로드한 ZIP의 MD5를 계산하고, 예상 MD5가 등록되어 있으면 비교합니다.
예상 MD5가 없으면 검증되지 않았음을 명확하게 출력합니다.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    "ai4i": {
        "url": "https://archive.ics.uci.edu/static/public/601/"
        "ai4i+2020+predictive+maintenance+dataset.zip",
        "expect": ["ai4i2020.csv"],
        # 2026-10-11 UCI 다운로드 원본과 직접 비교하여 확인한 MD5
        # UCI에서 별도로 공표한 공식 체크섬은 아님
        "md5": "f310bae04e9e66c45a63e18c3952fe7e",
    },
    "secom": {
        "url": "https://archive.ics.uci.edu/static/public/179/secom.zip",
        "expect": ["secom.data", "secom_labels.data", "secom.names"],
        # 2026-10-11 UCI 다운로드 원본과 직접 비교하여 확인한 MD5
        # UCI에서 별도로 공표한 공식 체크섬은 아님
        "md5": "a23a3fac2897c747ff9c8811ff980f5a",
    },
}


# ----------------------------------------------------------------------
# MD5 계산
# ----------------------------------------------------------------------
def calculate_md5(path: Path) -> str:
    """
    파일을 일정 크기씩 읽어 MD5 해시값을 계산합니다.

    전체 파일을 메모리에 한 번에 올리지 않도록
    청크 단위로 처리합니다.
    """

    md5 = hashlib.md5()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


# ----------------------------------------------------------------------
# 예상 MD5 검증
# ----------------------------------------------------------------------
def verify_md5(path: Path, expected_md5: str | None) -> bool:
    """
    실제 MD5와 예상 MD5를 비교합니다.

    예상값이 등록되지 않은 경우:
        검증되지 않았음을 출력하고 True를 반환합니다.

    예상값이 등록된 경우:
        일치하면 True, 불일치하면 False를 반환합니다.

    True는 반드시 검증 완료를 의미하는 것은 아닙니다.
    예상값이 없는 경우에는 검증을 생략한 상태입니다.
    """

    actual_md5 = calculate_md5(path)

    print(f"       md5={actual_md5}")

    if expected_md5 is None:
        print("[WARN] 예상 MD5가 등록되지 않아 일치 여부를 검증하지 못했습니다.")
        return True

    if actual_md5.lower() != expected_md5.lower():
        print("[FAIL] MD5 불일치")
        print(f"       예상: {expected_md5}")
        print(f"       실제: {actual_md5}")
        return False

    print("[PASS] MD5 일치")
    return True


# ----------------------------------------------------------------------
# ZIP 무결성 검사
# ----------------------------------------------------------------------
def verify_zip(path: Path) -> bool:
    """
    ZIP 파일이 정상적으로 열리고 내부 CRC 검사를 통과하는지 확인합니다.

    CRC 검사는 ZIP 파일 내부의 손상을 확인하는 검사이며,
    신뢰할 수 있는 예상 MD5와의 비교를 대체하지 않습니다.
    """

    try:
        with zipfile.ZipFile(path) as z:
            bad_file = z.testzip()

            if bad_file is not None:
                print(f"[FAIL] ZIP 내부 파일 손상: {bad_file}")
                return False

    except (zipfile.BadZipFile, OSError, RuntimeError) as e:
        print(f"[FAIL] ZIP 검사 실패: {type(e).__name__}: {e}")
        return False

    print("[PASS] ZIP 무결성 검사")
    return True


# ----------------------------------------------------------------------
# 공개 데이터셋 다운로드
# ----------------------------------------------------------------------
def fetch(name: str, spec: dict) -> bool:
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)

    zpath = RAW / f"{name}.zip"
    expected_md5 = spec.get("md5")

    # 기존에 압축 해제된 데이터가 있는 경우
    if all((out / f).exists() for f in spec["expect"]):
        print(f"[skip] {name} — 이미 있습니다")

        # ZIP이 남아 있다면 MD5와 무결성을 검사합니다.
        if zpath.exists():
            if not verify_md5(zpath, expected_md5):
                return False

            if not verify_zip(zpath):
                return False

        else:
            print("[WARN] 원본 ZIP이 없어 MD5 검증을 수행할 수 없습니다.")

        # 예상 MD5가 없거나 ZIP이 없다면 완전한 검증은 아님
        return True

    print(f"[get ] {name} ← {spec['url']}")

    try:
        r = requests.get(
            spec["url"],
            timeout=120,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()

    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        print("       UCI 서버가 가끔 느립니다. 잠시 후 다시 시도하거나")
        print("       브라우저로 직접 받아 data/raw/ 아래에 푸세요.")
        return False

    zpath.write_bytes(r.content)

    print(f"       {len(r.content):,} bytes")

    # 예상 MD5가 있다면 압축 해제 전에 반드시 비교합니다.
    if not verify_md5(zpath, expected_md5):
        return False

    # ZIP 내부 파일이 정상인지 검사합니다.
    if not verify_zip(zpath):
        return False

    # 압축 해제 경로가 지정된 데이터 폴더 밖으로 벗어나지 않도록 검사
    with zipfile.ZipFile(zpath) as z:
        destination = out.resolve()

        for member in z.infolist():
            target = (out / member.filename).resolve()

            if not target.is_relative_to(destination):
                print(f"[FAIL] 안전하지 않은 ZIP 경로: {member.filename}")
                return False

        z.extractall(out)

    missing = [f for f in spec["expect"] if not (out / f).exists()]

    if missing:
        print(f"[WARN] 기대한 파일이 없습니다: {missing}")
        print(f"       실제 내용: {[p.name for p in out.iterdir()]}")
        return False

    print(f"[ok  ] {name} → {out}")
    return True


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)

    # all(generator)는 첫 실패에서 실행을 중단하므로
    # 모든 데이터셋의 결과를 확인할 수 있도록 각각 실행
    results = [fetch(name, spec) for name, spec in SOURCES.items()]

    ok = all(results)

    if ok:
        print("\n전부 준비됐습니다. 이제 analysis/ 의 스크립트를 돌리면 됩니다.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------------------
# [구현 핵심]
# ----------------------------------------------------------------------

# UCI 공개 데이터셋 AI4I와 SECOM을 다운로드하는 스크립트 구현
# 데이터셋별 URL과 기대 파일명을 SOURCES에 등록해 관리하도록 구성
# 다운로드한 ZIP의 MD5를 계산하고 출력하도록 구현
# 기존 데이터가 존재하면 불필요한 다운로드를 생략하도록 구현

# 파일을 청크 단위로 읽는 MD5 계산 함수를 추가함
# AI4I와 SECOM의 로컬 ZIP을 UCI 서버 원본과 직접 비교하여 MD5 일치 확인
# 2026-10-11 확인한 UCI 제공 ZIP의 MD5를 기준값으로 등록
# 다운로드 및 기존 ZIP 검사 시 실제 MD5와 기준값을 비교하도록 개선
# MD5 불일치 파일을 거부해 데이터셋의 무결성 검증 강화
# 예상 MD5가 등록되지 않은 경우 검증되지 않았음을 경고하도록 구현
# ZIP 내부 CRC 검사로 손상된 압축파일을 탐지하도록 추가
# 기존 ZIP이 남아 있는 경우 재다운로드 없이 검사하도록 개선
# 압축 해제 경로가 대상 폴더를 벗어나지 않도록 검증을 추가함
# 한 데이터셋의 다운로드 실패 여부와 관계없이 다른 데이터셋도 검사하도록 변경
# AI4I와 SECOM 모두 등록된 MD5 일치 및 ZIP 무결성 검사 통과 확인

# 추후 보완 사항
# 1. UCI 데이터셋 갱신 시 기준 MD5 변경 이력과 버전 관리
# 2. 압축 해제된 개별 파일의 해시값을 검증하는 기능 추가
# 3. 다운로드 중단 시 임시 파일 정리 및 재시도 기능 추가
