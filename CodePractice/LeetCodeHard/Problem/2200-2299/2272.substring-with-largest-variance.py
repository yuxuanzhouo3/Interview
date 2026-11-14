"""
LC 2272. Substring With Largest Variance — Hard
https://leetcode.com/problems/substring-with-largest-variance/
Best: Enumerate char pair + Kadane with presence — O(26^2 n)
"""

class Solution:
    def largestVariance(self, s: str) -> int:
        from collections import Counter
        ans = 0
        freq = Counter(s)
        chars = list(freq)
        for a in chars:
            for b in chars:
                if a == b:
                    continue
                bal = 0
                has_b = False
                remain_b = freq[b]
                for ch in s:
                    if ch == a:
                        bal += 1
                    elif ch == b:
                        bal -= 1
                        has_b = True
                        remain_b -= 1
                    if has_b:
                        ans = max(ans, bal)
                    if bal < 0 and remain_b > 0:
                        bal = 0
                        has_b = False
        return ans


def _test():
    s = Solution()
    assert s.largestVariance("aababbb") == 3
    assert s.largestVariance("abcde") == 0
    print("LC 2272 OK")


if __name__ == "__main__":
    _test()
