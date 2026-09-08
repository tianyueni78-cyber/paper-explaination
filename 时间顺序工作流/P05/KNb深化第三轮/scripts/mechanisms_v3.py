"""KNb第三轮零解码控制核心；不引用A0，不计算目标值。"""

from collections import defaultdict


def k_candidates(coverage25, coverage50, opportunities25, opportunities50, minimum_gain):
    """A50只有带来足够新增覆盖或新增可行机会时才进入候选。"""
    marginal_coverage = max(0.0, coverage50 - coverage25)
    adds_opportunity = opportunities50 > opportunities25
    return (0.25, 0.5) if marginal_coverage >= minimum_gain or adds_opportunity else (0.25,)


def feasible_neighborhoods(opportunities):
    """用动作前合法机会数掩码N1—N6。"""
    return tuple(n for n in range(1, 7) if opportunities.get(n, 0) > 0)


def next_budget_tranche(spent, recent_batches, remaining, minimum_gain_per_decode, maximum=12, tranche=4):
    """先给4次；之后仅在最近一批单位解码收益达标时继续。"""
    if spent >= maximum or remaining <= 0:
        return 0
    if spent and not recent_batches:
        return 0
    if spent:
        gain, decodes = recent_batches[-1]
        if decodes <= 0 or gain / decodes < minimum_gain_per_decode:
            return 0
    return min(tranche, maximum - spent, remaining)


class FactorizedQ:
    """QK＋QN＋Qb＋QKN＋QNb；不含QKb。"""

    def __init__(self):
        self.terms = defaultdict(float)

    @staticmethod
    def _keys(state, action):
        k, n, b = action
        return (
            ("K", state, k),
            ("N", state, n),
            ("b", state, b),
            ("KN", state, k, n),
            ("Nb", state, n, b),
        )

    def value(self, state, action):
        return sum(self.terms[key] for key in self._keys(state, action))

    def update(self, state, action, reward, next_state, next_actions, alpha, gamma):
        future = max((self.value(next_state, candidate) for candidate in next_actions), default=0.0)
        td_error = reward + gamma * future - self.value(state, action)
        increment = alpha * td_error / 5
        for key in self._keys(state, action):
            self.terms[key] += increment
        return td_error


def confidence_action(values, minimum_gap):
    """至少两个动作且最优Q值领先达到阈值时才启用自适应。"""
    if len(values) < 2:
        return None
    ranked = sorted(values, key=lambda action: (-values[action], action))
    if values[ranked[0]] - values[ranked[1]] < minimum_gap:
        return None
    return ranked[0]


def state_change_score(old_bottlenecks, new_bottlenecks, old_opportunities, new_opportunities):
    """返回瓶颈集合变化与机会向量变化中的较大者。"""
    union = old_bottlenecks | new_bottlenecks
    jaccard_change = 1.0 - (len(old_bottlenecks & new_bottlenecks) / len(union) if union else 1.0)
    scale = max(sum(abs(value) for value in old_opportunities),
                sum(abs(value) for value in new_opportunities), 1)
    opportunity_change = sum(abs(left - right) for left, right in zip(old_opportunities, new_opportunities)) / scale
    return max(jaccard_change, opportunity_change)


def decay_state_credit(table, state, factor):
    """只衰减指定旧状态的信用。"""
    for key in list(table.terms):
        if key[1] == state:
            table.terms[key] *= factor
