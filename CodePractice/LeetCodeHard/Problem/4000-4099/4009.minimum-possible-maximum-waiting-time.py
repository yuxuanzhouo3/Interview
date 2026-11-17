"""
LC 4009. Minimum Possible Maximum Waiting Time — Hard
https://leetcode.com/problems/minimum-possible-maximum-waiting-time/
Best: maximize served prefix then minimize max wait; DP on two-pump fuel/time states
"""
INF = 10**9


class Solution:
    def minMaxWaitingTime(self, demand: list[int], fuel: list[int]) -> int:
        if demand[0] > max(fuel[0], fuel[1]):
            return -1
        states = 51 * 21 * 21

        def encode(f1, t1, t2):
            return (f1 * 21 + t1) * 21 + t2

        def decode(state, fuel_sum, used):
            t2 = state % 21
            state //= 21
            t1 = state % 21
            f1 = state // 21
            f2 = fuel_sum - used - f1
            return f1, f2, t1, t2

        fuel_sum = fuel[0] + fuel[1]
        used = 0
        dp = [INF] * states
        init = encode(fuel[0], 0, 0)
        dp[init] = 0
        cur = [init]
        for d in demand:
            dpp = [INF] * states
            nxt = []
            for state in cur:
                f1, f2, t1, t2 = decode(state, fuel_sum, used)
                if f1 >= d:
                    ns = encode(f1 - d, d, max(0, t2 - t1))
                    nw = max(dp[state], t1)
                    if dpp[ns] == INF:
                        nxt.append(ns)
                    dpp[ns] = min(dpp[ns], nw)
                if f2 >= d:
                    ns = encode(f1, max(0, t1 - t2), d)
                    nw = max(dp[state], t2)
                    if dpp[ns] == INF:
                        nxt.append(ns)
                    dpp[ns] = min(dpp[ns], nw)
            if not nxt:
                break
            used += d
            dp, cur = dpp, nxt
        return min((dp[s] for s in cur), default=INF)


def _test():
    s = Solution()
    assert s.minMaxWaitingTime([6, 8, 4, 6, 5], [16, 13]) == 6
    assert s.minMaxWaitingTime([10, 15], [12, 17]) == 0
    assert s.minMaxWaitingTime([10, 5], [8, 8]) == -1
    print("LC 4009 OK")


if __name__ == "__main__":
    _test()
