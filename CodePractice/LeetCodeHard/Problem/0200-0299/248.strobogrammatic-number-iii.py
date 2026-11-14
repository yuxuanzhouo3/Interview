"""
LC 248. Strobogrammatic Number III — Hard
https://leetcode.com/problems/strobogrammatic-number-iii/
Best: 按长度 DFS 生成镜像数并区间计数 — O(5^{L/2})
"""


class Solution:
    def strobogrammaticInRange(self, low: str, high: str) -> int:
        pairs = [("0", "0"), ("1", "1"), ("6", "9"), ("8", "8"), ("9", "6")]
        self.ans = 0
        lo, hi = low, high

        def dfs(path: list, left: int, right: int):
            if left > right:
                s = "".join(path)
                if len(s) == len(lo) and s < lo:
                    return
                if len(s) == len(hi) and s > hi:
                    return
                # 禁止前导零（单字符除外）
                if len(s) > 1 and s[0] == "0":
                    return
                self.ans += 1
                return
            for a, b in pairs:
                path[left], path[right] = a, b
                if left == right and a != b:
                    continue
                dfs(path, left + 1, right - 1)

        for length in range(len(lo), len(hi) + 1):
            dfs([""] * length, 0, length - 1)
        return self.ans


def _test():
    s = Solution()
    assert s.strobogrammaticInRange("50", "100") == 3  # 69, 88, 96
    assert s.strobogrammaticInRange("0", "0") == 1
    assert s.strobogrammaticInRange("0", "10") == 3  # 0,1,8
    assert s.strobogrammaticInRange("1", "1") == 1
    print("LC 248 OK")


if __name__ == "__main__":
    _test()
