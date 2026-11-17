"""
LC 899. Orderly Queue — Hard
Best: k==1 rotations min; else sort — O(n log n)
"""

class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        if k == 1:
            return min(s[i:] + s[:i] for i in range(len(s)))
        return "".join(sorted(s))


def _test():
    s = Solution()
    assert s.orderlyQueue("cba", 1) == "acb"
    assert s.orderlyQueue("baaca", 3) == "aaabc"
    print("LC 899 OK")


if __name__ == "__main__":
    _test()
