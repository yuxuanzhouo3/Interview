"""
8. Hard 贪心综合：排序键 + 堆
App: LC 871 最低加油次数；1834 单线程 CPU 思想。
"""

import heapq


def min_refuel_stops(target, start_fuel, stations):
    """LC 871: stations[i]=[pos, fuel]，可过站不加油，回头用堆取最大油。"""
    pq = []
    ans = prev = 0
    fuel = start_fuel
    stations = stations + [[target, 0]]
    for pos, f in stations:
        fuel -= pos - prev
        while fuel < 0 and pq:
            fuel += -heapq.heappop(pq)
            ans += 1
        if fuel < 0:
            return -1
        heapq.heappush(pq, -f)
        prev = pos
    return ans


def get_order(tasks):
    """LC 1834: tasks[i]=[enqueue, process]，返回处理顺序下标。"""
    indexed = sorted((e, p, i) for i, (e, p) in enumerate(tasks))
    res, pq = [], []
    t = i = 0
    n = len(tasks)
    while len(res) < n:
        while i < n and indexed[i][0] <= t:
            e, p, idx = indexed[i]
            heapq.heappush(pq, (p, idx))
            i += 1
        if pq:
            p, idx = heapq.heappop(pq)
            res.append(idx)
            t += p
        else:
            t = indexed[i][0]
    return res


def _test():
    assert min_refuel_stops(100, 10, [[10, 60], [20, 30], [30, 30], [60, 40]]) == 2
    assert get_order([[1, 2], [2, 4], [3, 2], [4, 1]]) == [0, 2, 3, 1]
    print("hard_greedy_apps OK")


APPLICATION = """
- 871: 走到哪油不够，从经过的站里选油最多的加（反悔堆）。
- 1834: 时间线 + 最短任务优先堆。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
