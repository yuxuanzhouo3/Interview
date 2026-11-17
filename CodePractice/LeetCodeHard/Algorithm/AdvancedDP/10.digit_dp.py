"""
10. 数位 DP · 范围内计数 / 价值总和
对照: 灵茶山 DP 题单 §十
App: LC 233 / 1012 / 2376 / 902 / 600
合并自: digit_dp.py；含题单 v2.1 上下界模板骨架
"""

from functools import lru_cache


def count_digit_one(n):
    """1..n 中数字 1 出现次数。LC 233。"""
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
    """[0,n] 各位互不相同个数。同 2376 思想。"""
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
    """通用骨架：pred(digits) 判断完整数字；返回 0..n 合法个数。"""
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


def digit_dp_range_count_zeros(low: int, high: int, target: int) -> int:
    """题单 v2.1：[low,high] 中恰好含 target 个 0 的整数个数（区分前导零）。"""
    low_s = list(map(int, str(low)))
    high_s = list(map(int, str(high)))
    n = len(high_s)
    diff_lh = n - len(low_s)

    @lru_cache(None)
    def dfs(i: int, cnt0: int, limit_low: bool, limit_high: bool) -> int:
        if cnt0 > target:
            return 0
        if i == n:
            return 1 if cnt0 == target else 0
        lo = low_s[i - diff_lh] if limit_low and i >= diff_lh else 0
        hi = high_s[i] if limit_high else 9
        res = 0
        start = lo
        if limit_low and i < diff_lh:
            res = dfs(i + 1, 0, True, False)
            start = 1
        for d in range(start, hi + 1):
            res += dfs(
                i + 1,
                cnt0 + (1 if d == 0 else 0),
                limit_low and d == lo,
                limit_high and d == hi,
            )
        return res

    return dfs(0, 0, True, True)


def _test():
    assert count_digit_one(13) == 6
    assert count_digit_one(0) == 0
    assert count_numbers_with_unique_digits_upto(20) == 20
    assert digit_dp_count(10, lambda digs: int("".join(map(str, digs))) % 2 == 0) == 6
    assert digit_dp_range_count_zeros(0, 10, 1) == 2  # 10, 0? 题单示例 digitDP(0,10,1)==2
    print("10.digit_dp OK")


APPLICATION = """
- 状态: pos / tight(贴上界) / lead_zero / 附加维。
- [L,R]=f(R)-f(L-1)；上下界一起枚举时 limit_low/high 为真不必记忆化。
- 前导零通常不计入「出现过的数字」。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
