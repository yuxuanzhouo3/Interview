"""
LC 440. K-th Smallest in Lexicographical Order — Hard
Best: Digit trie prefix count — O(log n * log n)
"""

class Solution:
    def findKthNumber(self, n: int, k: int) -> int:
        def count(p):
            steps = 0
            first, last = p, p
            while first <= n:
                steps += min(n, last) - first + 1
                first *= 10
                last = last * 10 + 9
            return steps

        cur = 1
        k -= 1
        while k:
            steps = count(cur)
            if steps <= k:
                cur += 1
                k -= steps
            else:
                cur *= 10
                k -= 1
        return cur


def _test():
    s = Solution()
    assert s.findKthNumber(13, 2) == 10
    assert s.findKthNumber(1, 1) == 1
    print("LC 440 OK")


if __name__ == "__main__":
    _test()
