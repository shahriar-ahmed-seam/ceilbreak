# Specification: Canonical RFC CeilBreak-001

## 1. Scope and Purpose

This document specifies the canonical protocols, schemas, invariants, and validation rules governing CeilBreak. Adherence to these data structures is mandatory for cross-engine compatibility with external inference and training harnesses (e.g., vLLM, DeepSeek, OpenRLHF, TRL).

---

## 2. Protocol Schemas and Field Definitions

### 2.1 Trajectory Record (`Trajectory`)

Represents a single rollout generated from an active or historical policy $\pi_\theta$.

| Field | Type | Required | Description |
|---|---|---|---|
| `trajectory_id` | `string` | Mandatory | Unique canonical identifier formatted as `step_{s}_{prompt_id}_rollout_{idx}`. |
| `prompt_id` | `string` | Mandatory | Identifier of the originating prompt. |
| `prompt_text` | `string` | Mandatory | Input text prompt presented to the model. |
| `completion_text` | `string` | Mandatory | Generated model output including reasoning tokens. |
| `tokens` | `List[string]` | Optional | Tokenized representation of `completion_text`. |
| `token_logprobs` | `List[float]` | Optional | Per-token log-probabilities under sampling policy $\pi_{\text{old}}$. |
| `ref_logprobs` | `List[float]` | Optional | Per-token log-probabilities under reference policy $\pi_{\text{ref}}$. |
| `reward_signal` | `RewardSignal` | Mandatory | Verification output schema. |
| `policy_iteration` | `integer` | Mandatory | Training iteration when rollout was generated ($\ge 0$). |
| `generation_entropy` | `float` | Optional | Mean per-token Shannon entropy ($\ge 0.0$). |
| `metadata` | `Dict[str, Any]` | Optional | Extensible domain metadata dictionary. |

### 2.2 Reward Signal Record (`RewardSignal`)

Represents structured verification metrics returned by automated verifiers.

| Field | Type | Range | Description |
|---|---|---|---|
| `binary_reward` | `integer` | `{0, 1}` | Strict binary pass/fail verification indicator. |
| `fractional_reward` | `float` | `[0.0, 1.0]` | Continuous partial credit score (e.g. unit test pass fraction). |
| `unit_tests_passed` | `integer` | `[0, total]` | Count of unit test assertions passed. |
| `total_unit_tests` | `integer` | `[0, inf)` | Total assertions evaluated in verification harness. |
| `syntax_valid` | `boolean` | `true/false` | Abstract syntax tree parse validation indicator. |
| `reasoning_length` | `integer` | `[0, inf)` | Token count inside reasoning trace (`<think>...</think>`). |
| `verifier_feedback` | `string` | N/A | Diagnostic trace or error message emitted by verifier. |

### 2.3 Rollout Group Record (`RolloutGroup`)

Groups candidate rollouts for a single prompt to calculate group-level baselines in GRPO.

| Field | Type | Invariant | Description |
|---|---|---|---|
| `group_id` | `string` | Unique | Identifier of rollout group. |
| `prompt` | `PromptItem` | Valid | Source prompt item. |
| `trajectories` | `List[Trajectory]` | `len >= 2` | Candidate rollouts for baseline computation. |
| `group_size` | `integer` | `== len(trajectories)` | Number of candidates sampled. |
| `empirical_pass_rate` | `float` | `[0.0, 1.0]` | Ratio of trajectories with `binary_reward == 1`. |
| `reward_mean` | `float` | `[0.0, 1.0]` | Empirical arithmetic mean of `fractional_reward`. |
| `reward_std` | `float` | `[0.0, 1.0]` | Empirical standard deviation of rewards. |

### 2.4 Advantage Estimate (`AdvantageEstimate`)

Stores group-normalized relative surprise advantages.

| Field | Type | Description |
|---|---|---|
| `trajectory_id` | `string` | Target trajectory identifier. |
| `prompt_id` | `string` | Associated prompt identifier. |
| `raw_reward` | `float` | Unnormalized reward value. |
| `group_mean` | `float` | Group baseline mean reward $\bar{r}$. |
| `group_std` | `float` | Group baseline standard deviation $\sigma_r$. |
| `advantage` | `float` | Relative advantage: $(r_i - \bar{r}) / (\sigma_r + \epsilon)$. |
| `is_easy_failure` | `boolean` | `true` if $\bar{r} \ge 0.70$ and $r_i < 0.50$. |
| `is_hard_win` | `boolean` | `true` if $\bar{r} \le 0.30$ and $r_i \ge 0.70$. |

---

## 3. Invariants and Validation Constraints

1. **Zero-Variance Guard**: When all rollouts within a group produce identical rewards ($\sigma_r < 10^{-6}$), the advantage for all trajectories in the group MUST evaluate to identically $0.0$.
2. **Entropy Non-Negativity**: Token Shannon entropy $\mathcal{H}$ MUST satisfy $\mathcal{H} \ge 0.0$.
3. **Clamping Bounds**: Clipped surrogate ratios in GRPO MUST be constrained to $[1 - \epsilon_{\text{clip}}, 1 + \epsilon_{\text{clip}}]$, where default $\epsilon_{\text{clip}} = 0.20$.
4. **Transition Gate Monotonicity**: Once the state transitions to `GRPO_ONLINE`, backward transition to `REST_EM_WARMUP` is prohibited in automated mode.
