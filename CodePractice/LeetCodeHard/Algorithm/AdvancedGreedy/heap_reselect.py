"""
4. 堆贪心 / 增幅再选
App: LC 1354 / 502 IPO。
"""

import heapq


def is_possible(target):
    """LC 1354: 从全 1 反复把某元素换成数组和，能否到 target。"""
    if len(target) == 1:
        return target[0] == 1
    total = sum(target)
    pq = [-x for x in target]
    heapq.heapify(pq)
    while True:
        cur = -heapq.heappop(pq)
        rest = total - cur
        if cur == 1 or rest == 1:
            return True
        if cur < rest or rest == 0 or cur % rest == 0:
            return False
        cur %= rest
        total = rest + cur
        heapq.heappush(pq, -cur)


def find_maximized_capital(k, w, profits, capital):
    """LC 502 IPO。"""
    projects = sorted(zip(capital, profits))
    i, n = 0, len(projects)
    pq = []
    for _ in range(k):
        while i < n and projects[i][0] <= w:
            heapq.heappush(pq, -projects[i][1])
            i += 1
        if not pq:
            break
        w += -heapq.heappop(pq)
    return w


def _test():
    assert is_possible([9, 3, 5]) is True
    assert is_possible([1, 1, 1, 2]) is False
    assert find_maximized_capital(2, 0, [1, 2, 3], [0, 1, 1]) == 4
    print("heap_reselect OK")


APPLICATION = """
- 每次取全局最优增量：大根堆。
- 1354 逆向从 target 还原更高效。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
