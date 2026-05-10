"""
LLM AAI 단독 호출 테스트.

환경 변수에 ANTHROPIC_API_KEY를 설정한 뒤 실행:
    export ANTHROPIC_API_KEY=sk-...
    python tools/test_llm_call.py

목적:
- 실제 LLM이 우리 prompt에 대해 schema에 맞는 JSON을 잘 반환하는지 확인
- prompt 변경 후 회귀 테스트

이 스크립트를 evaluation 파이프라인의 일부로 만들지는 않음 (선택적 검증용).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from onpolicy.envs.uav.uav_env import UAVTrackingEnv
from tools.llm_aai import (
    LLMAAI, _extract_telemetry, _make_user_prompt,
    SYSTEM_PROMPT, heuristic_aai_output,
)


def main():
    # 1) 환경 만들고 한 step 진행
    env = UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=True)
    env.seed(42)
    env.reset()
    for _ in range(10):
        actions = np.random.uniform(-1, 1, (5, 2)).astype(np.float32)
        env.step(actions)

    # 2) Telemetry 추출 + prompt 출력
    telemetry = _extract_telemetry(env)
    user_prompt = _make_user_prompt(telemetry)
    print("=" * 70)
    print("SYSTEM PROMPT:")
    print("=" * 70)
    print(SYSTEM_PROMPT)
    print("\n" + "=" * 70)
    print("USER PROMPT:")
    print("=" * 70)
    print(user_prompt)

    # 3) Heuristic 결과 (비교 baseline)
    h_out = heuristic_aai_output(env)
    print("\n" + "=" * 70)
    print("HEURISTIC OUTPUT (fallback):")
    print("=" * 70)
    import json
    print(json.dumps(h_out, indent=2))

    # 4) 실제 LLM 호출 (API 키 있을 때만)
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n[skip] ANTHROPIC_API_KEY not set; skipping live LLM call.")
        return

    print("\n" + "=" * 70)
    print("LIVE LLM CALL:")
    print("=" * 70)
    aai = LLMAAI(provider='anthropic', call_every=1, verbose=True)
    text = aai._call_llm_api(SYSTEM_PROMPT, user_prompt)
    print("Raw response:")
    print(text)

    parsed = aai._parse_llm_response(
        text,
        expected_n_targets=env.num_targets,
        expected_n_uavs=env.num_uavs,
    )
    print("\nParsed (validated) output:")
    print(json.dumps(parsed, indent=2))

    # 통계
    print(f"\nStats: {aai.stats}")


if __name__ == '__main__':
    main()
