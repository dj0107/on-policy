"""
llm_aai.py
==========
LLM (Anthropic Claude) 기반 AAI 모듈.

설계 원칙:
1. **호출 빈도 throttling**: 매 step LLM 호출은 비용/시간 모두 비현실적.
   기본 매 K=10 step마다 호출. 그 사이엔 마지막 출력 재사용.
2. **상태 캐싱**: 동일 상태(반올림된 telemetry)가 들어오면 캐시 반환.
   evaluation에서 reset 시드가 같으면 trajectory가 비슷해 cache hit 비율 높음.
3. **JSON schema 강제**: LLM에 system prompt로 출력 스키마 명시 + 파싱 후 validation.
4. **Fallback**: API 호출 실패 / JSON parsing 실패 시 heuristic AAI로 fallback.
5. **Stateless callback**: 환경 측이 호출하는 함수는 stateless해야 multi-thread 안전.
   따라서 LLMAAI 클래스 인스턴스가 callback의 closure 안에서 관리됨.

Usage (evaluate_trained.py에서):
    --aai_callback tools.llm_aai:default_callback
    
    또는 직접:
    from tools.llm_aai import LLMAAI
    aai = LLMAAI(model='claude-sonnet-4-6', call_every=10)
    env = UAVTrackingEnv(num_uavs=5, num_targets=2, aai_callback=aai.callback)

API 키:
    환경 변수 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY (provider에 따라).
    없으면 heuristic으로만 동작 (LLM 호출 0회).
"""
import os
import json
import time
import hashlib
import warnings
from typing import Optional, Callable, Dict, Any
import numpy as np


# ============================================================================
# 출력 스키마 (JSON으로 LLM이 반환해야 하는 형식)
# ============================================================================
AAI_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["targets", "uavs"],
    "properties": {
        "targets": {
            "type": "array",
            "description": "타겟별 AAI 출력 — 인덱스 순서 == env.targets",
            "items": {
                "type": "object",
                "required": ["W_kt", "epsilon_kt"],
                "properties": {
                    "W_kt": {"type": "number", "minimum": 0.5, "maximum": 5.0},
                    "epsilon_kt": {"type": "number", "minimum": 0.5, "maximum": 20.0}
                }
            }
        },
        "uavs": {
            "type": "array",
            "description": "UAV별 송신전력 — 인덱스 순서 == env.uavs",
            "items": {
                "type": "object",
                "required": ["p_ut"],
                "properties": {
                    "p_ut": {"type": "number", "minimum": 5.0, "maximum": 30.0}
                }
            }
        }
    }
}


SYSTEM_PROMPT = """You are a high-level reasoning module for a UAV swarm tracking
ground targets. Your job: given the current swarm state, output a small JSON
with three control parameters per target/UAV:

- W_{k,t} (priority weight) ∈ [0.5, 5.0] — higher means tracking accuracy of this
  target is more important (e.g. close to a risk zone or slow-moving).
- epsilon_{k,t} (tracking accuracy threshold) ∈ [0.5, 20.0] — lower means we want
  tighter PCRLB on this target.
- p_{u,t} (UAV transmit power, W) ∈ [5.0, 30.0] — higher gives stronger sensing
  but uses more energy.

Decision principles:
- Targets near a risk zone (small d_Z) should get HIGH W and LOW epsilon.
- UAVs far from their assigned target should use HIGHER p_ut to maintain SNR.
- UAVs with LOW remaining energy should use LOWER p_ut when possible.
- A UAV directly above its target needs only modest p_ut.

Output ONLY a single JSON object matching the schema. No commentary, no markdown."""


def _make_user_prompt(telemetry: Dict[str, Any]) -> str:
    """env에서 추출한 telemetry를 프롬프트로 변환"""
    lines = ["Current swarm telemetry:\n"]
    lines.append(f"Time step t = {telemetry['time_slot']}, dt = {telemetry['dt']}s")
    lines.append(f"Map bounds: [{telemetry['map_min']:.0f}, {telemetry['map_max']:.0f}] m")
    lines.append(f"BS at {telemetry['bs_pos']}\n")

    lines.append("Risk zones:")
    for i, cz in enumerate(telemetry['critical_zones']):
        lines.append(f"  Zone {i}: ({cz[0]:.0f}, {cz[1]:.0f})")
    lines.append("")

    lines.append(f"Targets ({len(telemetry['targets'])}):")
    for k, t in enumerate(telemetry['targets']):
        pos = t['est_pos']
        vel = t['est_vel']
        lines.append(
            f"  Target {k}: est_pos=({pos[0]:.0f},{pos[1]:.0f}), "
            f"est_vel=({vel[0]:.1f},{vel[1]:.1f}), "
            f"d_Z={t['d_Z']:.0f}m, F_kt={t['F_kt']:.2f}"
        )
    lines.append("")

    lines.append(f"UAVs ({len(telemetry['uavs'])}):")
    for u, uav in enumerate(telemetry['uavs']):
        pos = uav['pos']
        lines.append(
            f"  UAV {u}: pos=({pos[0]:.0f},{pos[1]:.0f}), "
            f"E_remaining={uav['energy']:.0f}/{uav['max_energy']:.0f}, "
            f"assigned_to=Target{uav['assigned']}, "
            f"distance_to_target={uav['dist_to_target']:.0f}m"
        )

    lines.append(
        f"\nReturn JSON with {len(telemetry['targets'])} target entries "
        f"and {len(telemetry['uavs'])} uav entries."
    )
    return "\n".join(lines)


def _extract_telemetry(env) -> Dict[str, Any]:
    """env에서 LLM 입력용 dict 추출"""
    targets = []
    for t in env.targets:
        targets.append({
            'est_pos': t.S_global[:2].tolist(),
            'est_vel': t.S_global[2:].tolist(),
            'd_Z': float(t.d_Z_kt),
            'F_kt': float(t.F_kt),
        })
    uavs = []
    for u_idx, uav in enumerate(env.uavs):
        t_idx = env.assignment.get(u_idx, 0)
        target = env.targets[t_idx]
        dist = float(np.linalg.norm(uav.pos - target.S_global[:2]))
        uavs.append({
            'pos': uav.pos.tolist(),
            'energy': float(uav.energy),
            'max_energy': float(uav.max_energy),
            'assigned': int(t_idx),
            'dist_to_target': dist,
        })
    return {
        'time_slot': int(env.time_slot),
        'dt': float(env.dt),
        'map_min': float(env.map_min),
        'map_max': float(env.map_max),
        'bs_pos': env.bs_pos.tolist(),
        'critical_zones': [cz.tolist() for cz in env.critical_zones],
        'targets': targets,
        'uavs': uavs,
    }


def _telemetry_cache_key(telemetry: Dict[str, Any]) -> str:
    """반올림된 telemetry로 cache key 생성 (작은 noise는 cache hit 유발)"""
    rounded = {
        'targets': [
            (round(t['est_pos'][0] / 30) * 30,  # 30m 단위
             round(t['est_pos'][1] / 30) * 30,
             round(t['d_Z'] / 50) * 50,          # 50m 단위
             ) for t in telemetry['targets']
        ],
        'uavs': [
            (round(u['pos'][0] / 30) * 30,
             round(u['pos'][1] / 30) * 30,
             u['assigned'],
             round(u['dist_to_target'] / 50) * 50,
             ) for u in telemetry['uavs']
        ],
    }
    return hashlib.md5(json.dumps(rounded, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================================
# Heuristic fallback (env._run_aai_heuristic과 동일 로직)
# ============================================================================
def heuristic_aai_output(env) -> Dict[str, Any]:
    """LLM 실패 시 fallback. env 상태로부터 직접 W/eps/p 결정."""
    targets_out = []
    for t in env.targets:
        if t.d_Z_kt < 100.0:
            W, eps = 3.0, 2.0
        elif t.d_Z_kt < 200.0:
            W, eps = 2.0, 5.0
        else:
            W, eps = 1.0, 5.0
        targets_out.append({'W_kt': W, 'epsilon_kt': eps})

    uavs_out = []
    for u_idx, uav in enumerate(env.uavs):
        t_idx = env.assignment.get(u_idx, 0)
        target = env.targets[t_idx]
        dist = np.linalg.norm(uav.pos - target.S_global[:2])
        if dist > 200.0:
            p = 25.0
        elif dist < 50.0:
            p = 10.0
        else:
            p = 15.0
        uavs_out.append({'p_ut': p})

    return {'targets': targets_out, 'uavs': uavs_out}


# ============================================================================
# LLMAAI 클래스
# ============================================================================
class LLMAAI:
    """LLM AAI manager. 인스턴스의 self.callback을 env.aai_callback에 넘김."""

    def __init__(
        self,
        model: str = 'claude-sonnet-4-6',
        provider: str = 'anthropic',
        call_every: int = 10,
        max_cache_size: int = 5000,
        verbose: bool = False,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """
        Args:
            model: LLM model id
            provider: 'anthropic' | 'openai' | 'mock' (mock = heuristic만 사용, no API)
            call_every: 매 K step마다 LLM 호출. 사이엔 마지막 출력 재사용.
                        1로 두면 매 step 호출 (비싸지만 정확).
            max_cache_size: telemetry hash → output 캐시 최대 크기
            timeout: API 호출 타임아웃 (초)
            max_retries: API 실패 시 재시도 횟수
        """
        self.model = model
        self.provider = provider
        self.call_every = max(1, int(call_every))
        self.max_cache_size = max_cache_size
        self.verbose = verbose
        self.timeout = timeout
        self.max_retries = max_retries

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_output: Optional[Dict[str, Any]] = None
        self._last_step: int = -10**9

        self._client = None
        if provider == 'anthropic':
            try:
                import anthropic
                api_key = os.environ.get('ANTHROPIC_API_KEY')
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
                else:
                    warnings.warn(
                        'ANTHROPIC_API_KEY not set; LLMAAI will use heuristic fallback only.'
                    )
            except ImportError:
                warnings.warn(
                    'anthropic package not installed; LLMAAI will use heuristic fallback only.'
                )
        elif provider == 'openai':
            try:
                import openai
                api_key = os.environ.get('OPENAI_API_KEY')
                if api_key:
                    self._client = openai.OpenAI(api_key=api_key)
                else:
                    warnings.warn(
                        'OPENAI_API_KEY not set; LLMAAI will use heuristic fallback only.'
                    )
            except ImportError:
                warnings.warn(
                    'openai package not installed; LLMAAI will use heuristic fallback only.'
                )
        elif provider == 'deepseek':
            try:
                import openai
                api_key = os.environ.get('DEEPSEEK_API_KEY')
                if api_key:
                    self._client = openai.OpenAI(
                        api_key=api_key,
                        base_url='https://api.deepseek.com',
                    )
                else:
                    warnings.warn(
                        'DEEPSEEK_API_KEY not set; LLMAAI will use heuristic fallback only.'
                    )
            except ImportError:
                warnings.warn(
                    'openai package not installed; LLMAAI will use heuristic fallback only.'
                )
        elif provider == 'gemini':
            try:
                import openai
                api_key = os.environ.get('GEMINI_API_KEY')
                if api_key:
                    self._client = openai.OpenAI(
                        api_key=api_key,
                        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
                    )
                else:
                    warnings.warn(
                        'GEMINI_API_KEY not set; LLMAAI will use heuristic fallback only.'
                    )
            except ImportError:
                warnings.warn(
                    'openai package not installed; LLMAAI will use heuristic fallback only.'
                )
        elif provider == 'mock':
            pass  # client 없음, heuristic만 사용
        else:
            raise ValueError(f'unknown provider: {provider}. '
                             f'Supported: anthropic, openai, deepseek, gemini, mock')

        # 통계
        self.stats = {
            'n_calls': 0, 'n_cache_hits': 0, 'n_throttled': 0,
            'n_api_calls': 0, 'n_api_failures': 0, 'n_parse_failures': 0,
            'total_api_time': 0.0,
        }

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """LLM API 호출. 실패 시 None 반환."""
        if self._client is None:
            return None

        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.time()
                if self.provider == 'anthropic':
                    msg = self._client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    text = msg.content[0].text
                elif self.provider in ('openai', 'deepseek', 'gemini'):
                    kwargs = dict(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        timeout=self.timeout,
                    )
                    # response_format JSON mode: openai/deepseek 지원, gemini는 미적용
                    if self.provider in ('openai', 'deepseek'):
                        kwargs['response_format'] = {"type": "json_object"}
                    resp = self._client.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content
                else:
                    return None
                self.stats['n_api_calls'] += 1
                self.stats['total_api_time'] += time.time() - t0
                return text
            except Exception as e:
                if self.verbose:
                    print(f'[LLMAAI] API call failed (attempt {attempt+1}): {e}')
                if attempt == self.max_retries:
                    self.stats['n_api_failures'] += 1
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    @staticmethod
    def _parse_llm_response(text: str, expected_n_targets: int,
                            expected_n_uavs: int) -> Optional[Dict[str, Any]]:
        """LLM 응답 텍스트에서 JSON 추출 + 검증"""
        if text is None:
            return None
        # 코드 블록 제거 (markdown으로 래핑한 경우)
        text = text.strip()
        if text.startswith('```'):
            # ```json ... ``` 또는 ``` ... ```
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # JSON이 아닌 텍스트가 섞여있을 수 있음 → { ... } 첫 매치
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        # 스키마 검증 (수동)
        if not isinstance(data, dict):
            return None
        if 'targets' not in data or 'uavs' not in data:
            return None
        targets = data['targets']
        uavs = data['uavs']
        if len(targets) != expected_n_targets or len(uavs) != expected_n_uavs:
            return None

        # 값 범위 클리핑 (LLM이 범위 벗어나도 안전하게)
        clean_targets = []
        for t in targets:
            if not isinstance(t, dict):
                return None
            W = float(t.get('W_kt', 1.0))
            eps = float(t.get('epsilon_kt', 5.0))
            clean_targets.append({
                'W_kt': float(np.clip(W, 0.5, 5.0)),
                'epsilon_kt': float(np.clip(eps, 0.5, 20.0)),
            })
        clean_uavs = []
        for u in uavs:
            if not isinstance(u, dict):
                return None
            p = float(u.get('p_ut', 15.0))
            clean_uavs.append({'p_ut': float(np.clip(p, 5.0, 30.0))})

        return {'targets': clean_targets, 'uavs': clean_uavs}

    def _apply_output(self, env, output: Dict[str, Any]):
        """LLM/heuristic output을 env에 주입"""
        for k, t_out in enumerate(output['targets']):
            env.targets[k].W_kt = float(t_out['W_kt'])
            env.targets[k].epsilon_kt = float(t_out['epsilon_kt'])
        for u, u_out in enumerate(output['uavs']):
            env.uavs[u].p_ut = float(u_out['p_ut'])

    def callback(self, env):
        """env.aai_callback에 등록할 함수.

        env._invoke_aai에서 호출됨. env에는 이미:
        - target.d_Z_kt가 갱신되어 있어야 함  (env._run_aai_heuristic가 함)
        - assignment가 갱신되어 있어야 함

        그러나 callback 모드에서는 env가 _run_aai_heuristic을 부르지 않으므로,
        d_Z_kt는 여기서 직접 계산.
        """
        # 0) 환경 dim 변경 감지 → cache 무효화 (UAV 수가 바뀐 새 env일 수 있음)
        cur_dim = (env.num_uavs, env.num_targets)
        if getattr(self, '_last_dim', None) != cur_dim:
            self._cache.clear()
            self._last_output = None
            self._last_step = -10**9
            self._last_dim = cur_dim

        # 1) d_Z_kt 갱신
        for target in env.targets:
            est_pos = target.S_global[:2]
            target.d_Z_kt = min(
                float(np.linalg.norm(est_pos - cz)) for cz in env.critical_zones
            )

        self.stats['n_calls'] += 1

        # 2) Throttling: 매 call_every step마다만 LLM 재호출
        steps_since_last = env.time_slot - self._last_step
        if self._last_output is not None and steps_since_last < self.call_every:
            self.stats['n_throttled'] += 1
            self._apply_output(env, self._last_output)
            return

        # 3) Cache lookup
        telemetry = _extract_telemetry(env)
        cache_key = _telemetry_cache_key(telemetry)
        if cache_key in self._cache:
            self.stats['n_cache_hits'] += 1
            output = self._cache[cache_key]
            self._apply_output(env, output)
            self._last_output = output
            self._last_step = env.time_slot
            return

        # 4) LLM 호출 (provider 있으면)
        output = None
        if self._client is not None:
            user_prompt = _make_user_prompt(telemetry)
            text = self._call_llm_api(SYSTEM_PROMPT, user_prompt)
            output = self._parse_llm_response(
                text,
                expected_n_targets=env.num_targets,
                expected_n_uavs=env.num_uavs,
            )
            if output is None:
                self.stats['n_parse_failures'] += 1

        # 5) Fallback: heuristic
        if output is None:
            output = heuristic_aai_output(env)

        # 6) Cache & apply
        if len(self._cache) >= self.max_cache_size:
            # 단순 FIFO eviction
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[cache_key] = output
        self._last_output = output
        self._last_step = env.time_slot
        self._apply_output(env, output)


# ============================================================================
# 편의 함수: argparse 한 줄로 callback 만들기
# ============================================================================
_default_instance: Optional[LLMAAI] = None


def default_callback(env):
    """evaluate_trained.py --aai_callback tools.llm_aai:default_callback 용.

    환경 변수로 LLM 설정:
        LLM_AAI_PROVIDER=anthropic    → ANTHROPIC_API_KEY 필요
        LLM_AAI_PROVIDER=openai       → OPENAI_API_KEY 필요
        LLM_AAI_PROVIDER=deepseek     → DEEPSEEK_API_KEY 필요
        LLM_AAI_PROVIDER=gemini       → GEMINI_API_KEY 필요
        LLM_AAI_PROVIDER=mock         → API 없이 heuristic만 사용

        LLM_AAI_MODEL=claude-sonnet-4-6   (anthropic 기본)
        LLM_AAI_MODEL=gpt-4o              (openai)
        LLM_AAI_MODEL=deepseek-chat       (deepseek)
        LLM_AAI_MODEL=gemini-2.0-flash    (gemini)

        LLM_AAI_CALL_EVERY=10   (매 K step마다 LLM 호출)
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = LLMAAI(
            model=os.environ.get('LLM_AAI_MODEL', 'claude-sonnet-4-6'),
            provider=os.environ.get('LLM_AAI_PROVIDER', 'anthropic'),
            call_every=int(os.environ.get('LLM_AAI_CALL_EVERY', '10')),
            verbose=os.environ.get('LLM_AAI_VERBOSE', '0') == '1',
        )
    _default_instance.callback(env)


def mock_callback(env):
    """LLM 호출 없이 heuristic만 사용 (테스트용). evaluate_trained 흐름 검증에 유용."""
    global _default_instance
    if _default_instance is None or _default_instance.provider != 'mock':
        _default_instance = LLMAAI(provider='mock', call_every=10)
    _default_instance.callback(env)
