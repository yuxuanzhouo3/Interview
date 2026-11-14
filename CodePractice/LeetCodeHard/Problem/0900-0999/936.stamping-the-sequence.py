"""
LC 936. Stamping The Sequence — Hard
https://leetcode.com/problems/stamping-the-sequence/
Best: Reverse greedy window peel — O((n-m+1)*m*n)
"""

from typing import List


class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        m, n = len(stamp), len(target)
        t = list(target)
        done = [False] * (n - m + 1)
        ans = []
        stars = 0

        def can_stamp(i):
            changed = False
            for j in range(m):
                if t[i + j] == "*":
                    continue
                if t[i + j] != stamp[j]:
                    return False
                changed = True
            return changed

        def do_stamp(i):
            nonlocal stars
            for j in range(m):
                if t[i + j] != "*":
                    t[i + j] = "*"
                    stars += 1

        while stars < n:
            progressed = False
            for i in range(n - m + 1):
                if not done[i] and can_stamp(i):
                    do_stamp(i)
                    done[i] = True
                    ans.append(i)
                    progressed = True
                    if stars == n:
                        break
            if not progressed:
                return []
        ans.reverse()
        return ans


def _test():
    s = Solution()
    r = s.movesToStamp("abc", "ababc")
    assert r in ([0, 2], [1, 0, 2]) or (len(r) <= 10 and True)
    # verify by simulation
    def apply(stamp, target, moves):
        n = len(target)
        cur = ["?"] * n
        for i in moves:
            for j, c in enumerate(stamp):
                cur[i + j] = c
        return "".join(cur) == target
    assert apply("abc", "ababc", r)
    r2 = s.movesToStamp("abca", "aabcaca")
    assert apply("abca", "aabcaca", r2)
    assert s.movesToStamp("a", "b") == []
    print("LC 936 OK")


if __name__ == "__main__":
    _test()
