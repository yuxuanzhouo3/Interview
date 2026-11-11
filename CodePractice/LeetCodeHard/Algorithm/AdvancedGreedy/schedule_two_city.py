"""
7. 两地调度 — 代价差排序贪心
App: LC 1029。
"""


def two_city_sched_cost(costs):
    """一半人去 A，一半去 B。"""
    n = len(costs) // 2
    costs = sorted(costs, key=lambda x: x[0] - x[1])
    # 前 n 个相对更便宜去 A
    return sum(c[0] for c in costs[:n]) + sum(c[1] for c in costs[n:])


def _test():
    costs = [[10, 20], [30, 200], [400, 50], [30, 20]]
    assert two_city_sched_cost(costs) == 110
    print("schedule_two_city OK", two_city_sched_cost(costs))


APPLICATION = """
- 按 costA-costB 排序：差值越小越该去 A。
- 推广：多维代价差、加权分配。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
