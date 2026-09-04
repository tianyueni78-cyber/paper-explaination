"""最小K/N/b控制；零解码纯函数，不引用baseline或未来目标。"""
from statistics import median


def rank_region(l, m, a, k):
    scale = max(l) or 1.0
    scores = [value / scale * (1 + int(machine) + int(agv))
              for value, machine, agv in zip(l, m, a)]
    return set(sorted(range(len(l)), key=lambda i: (-scores[i], i))[:k])

def state_id(t, m, alternatives, selected):
    transport = sum(t[i] for i in selected)
    wait = transport + sum(m[i] for i in selected)
    fraction = transport / wait if wait else 0.0
    alternative_fraction = sum(alternatives[i] for i in selected) / len(selected) if selected else 0.0
    return 2 * int(fraction > 0.5) + int(alternative_fraction > 0.5)


def credit(records):
    cost = sum(cost for _, cost in records)
    return sum(gain for gain, _ in records) / cost if cost else 0.0

def choose_n(history, visits, step, explore_cursor):
    for n, count in enumerate(visits):
        if count == 0:
            return n, explore_cursor
    if step % 4 == 0:
        return explore_cursor % 6, explore_cursor + 1
    return min(range(6), key=lambda n: (-credit(history.get(n, [])), n)), explore_cursor

def choose_b(history, n, remaining):
    records = history.get(n, [])
    values = [credit(rows) for rows in history.values() if rows]
    large = bool(records) and credit(records) > 0 and credit(records) >= median(values)
    return min(12 if large else 4, remaining)

def remember(history, n, gain, cost, window=3):
    if cost:
        rows = history.setdefault(n, [])
        rows.append((gain, cost))
        if window is not None:
            del rows[:-window]
