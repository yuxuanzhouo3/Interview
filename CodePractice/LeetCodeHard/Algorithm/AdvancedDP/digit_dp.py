"""
AdvancedDP / 数位 DP
按位填数字；状态含 pos / tight / lead_zero / 额外统计
App: LC 233 思想；范围内计数
"""

from functools import lru_cache


def count_digit_one(n):
    """统计 1..n 中数字 1 出现次数。LC 233。"""
    if n < 0:
        return 0
    s = str(n)

    @lru_cache(None)
    def dfs(pos, tight, ones, started):
        if pos == len(s):
            return ones
        up = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, up + 1):
            ntight = tight and d == up
            nstarted = started or d != 0
            nones = ones + (1 if nstarted and d == 1 else 0)
            total += dfs(pos + 1, ntight, nones, nstarted)
        return total

    return dfs(0, True, 0, False)


def count_numbers_with_unique_digits_upto(n):
    """统计 [0, n] 中各位数字互不相同的个数（含 0）。"""
    if n < 0:
        return 0
    s = str(n)

    @lru_cache(None)
    def dfs(pos, tight, mask, started):
        if pos == len(s):
            return 1
        up = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, up + 1):
            if started and (mask >> d) & 1:
                continue
            ntight = tight and d == up
            if not started and d == 0:
                total += dfs(pos + 1, ntight, mask, False)
            else:
                total += dfs(pos + 1, ntight, mask | (1 << d), True)
        return total

    return dfs(0, True, 0, False)


def digit_dp_count(n, pred):
    """通用骨架：pred(digits_tuple) 判断完整数字是否合法；返回 0..n 合法个数。"""
    if n < 0:
        return 0
    s = str(n)

    @lru_cache(None)
    def dfs(pos, tight, path):
        if pos == len(s):
            return 1 if pred(path) else 0
        up = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, up + 1):
            total += dfs(pos + 1, tight and d == up, path + (d,))
        return total

    return dfs(0, True, ())


def _test():
    assert count_digit_one(13) == 6  # 1,10,11,12,13 → 1出现6次
    assert count_digit_one(0) == 0
    assert count_numbers_with_unique_digits_upto(20) == 20  # 0..20 去掉 11 → 21-1
    # 偶数个数 0..10: 0,2,4,6,8,10
    assert digit_dp_count(10, lambda digs: int("".join(map(str, digs))) % 2 == 0) == 6
    print("digit_dp OK")


APPLICATION = """
- 核心状态: pos, tight(是否贴上界), lead_zero(是否仍前导零), 题目附加维。
- 写区间 [L,R] = f(R)-f(L-1)。
- 前导零时通常不记「出现过的数字」，避免 0012 误判重复。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
