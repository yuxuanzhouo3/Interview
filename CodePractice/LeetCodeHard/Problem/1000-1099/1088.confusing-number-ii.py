"""
LC 1088. Confusing Number II — Hard
https://leetcode.com/problems/confusing-number-ii/
Tags: Math, Backtracking
Best: 只生成可旋转数字并检查翻转后不同
"""

class Solution:
    def confusingNumberII(self, n: int) -> int:
        rotate = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        digits = [0, 1, 6, 8, 9]
        self.ans = 0

        def is_confusing(num: int) -> bool:
            x, rev = num, 0
            while x:
                d = x % 10
                rev = rev * 10 + rotate[d]
                x //= 10
            return rev != num

        def dfs(cur: int):
            if cur > n:
                return
            if cur != 0 and is_confusing(cur):
                self.ans += 1
            if cur == 0:
                for d in digits[1:]:
                    dfs(d)
            else:
                for d in digits:
                    nxt = cur * 10 + d
                    if nxt <= n:
                        dfs(nxt)

        dfs(0)
        return self.ans


def _test():
    sol = Solution()
    assert sol.confusingNumberII(20) == 6
    assert sol.confusingNumberII(100) == 19
    print("LC 1088 OK")


if __name__ == "__main__":
    _test()

