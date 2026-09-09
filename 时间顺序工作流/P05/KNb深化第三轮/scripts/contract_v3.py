"""第三轮开发协议防漂移校验。"""


EXPECTED_PROTOCOL = {
    "version": "knb3-development-v1",
    "scope": "development_only_not_confirmation",
    "instances": [f"Mk{i:02}" for i in range(1, 7)],
    "heldout_forbidden": ["Mk07", "Mk08", "Mk09", "Mk10"],
    "k_fractions": [0.25, 0.5],
    "neighborhoods": [1, 2, 3, 4, 5, 6],
    "budget_targets": [4, 8, 12],
    "budget_tranche": 4,
    "development_budget": 96,
    "development_parent_seed": 101,
    "development_repeats": [0, 1, 2],
    "timeout_seconds": 1800,
    "smoke_budget": 24,
    "maximum_decisions": 960,
    "attempts_per_decode": 40,
    "n1_opportunity_samples": 64,
    "n2_opportunity_samples": 64,
    "fallback_action": [0.25, 3, 4],
    "q_terms": ["K", "N", "b", "KN", "Nb"],
    "q_kb_enabled": False,
    "q_reset": "each_independent_run",
    "alpha": 0.1,
    "gamma": 0.9,
    "k_minimum_coverage_gain": 0.05,
    "continue_minimum_gain_per_decode": 1e-12,
    "confidence_minimum_gap": 0.000001,
    "state_change_threshold": 0.5,
    "credit_decay_factor": 0.5,
    "confidence_gate": "training_fold_calibrated_action_gap",
    "fixed_generation_ratio": None,
    "true_decode_charging": "every_complete_decode",
    "performance_claims_from_unit_tests": False,
    "validation_data_opened": False,
}


def validate_protocol(protocol):
    if protocol != EXPECTED_PROTOCOL:
        raise ValueError("第三轮协议与已确认机制不一致，禁止运行")
