"""
LC 854. K-Similar Strings — Hard
Best: BFS swap mismatches — O(n! /..) pruned
"""

from collections import deque


class Solution:
    def kSimilarity(self, s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        s1 = list(s1)
        # only positions differ
        A, B = [], []
        for a, b in zip(s1, s2):
            if a != b:
                A.append(a)
                B.append(b)
        start = "".join(A)
        target = "".join(B)
        q = deque([(start, 0)])
        seen = {start}
        n = len(start)
        while q:
            cur, d = q.popleft()
            # find first mismatch
            i = 0
            while i < n and cur[i] == target[i]:
                i += 1
            for j in range(i + 1, n):
                if cur[j] == target[i] and cur[j] != target[j]:
                    arr = list(cur)
                    arr[i], arr[j] = arr[j], arr[i]
                    nxt = "".join(arr)
                    if nxt == target:
                        return d + 1
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, d + 1))
        return -1


def _test():
    s = Solution()
    assert s.kSimilarity("ab", "ba") == 1
    assert s.kSimilarity("abc", "bca") == 2
    print("LC 854 OK")


if __name__ == "__main__":
    _test()
