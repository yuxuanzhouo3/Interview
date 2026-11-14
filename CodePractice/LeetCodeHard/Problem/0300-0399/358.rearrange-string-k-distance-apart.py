"""
LC 358. Rearrange String k Distance Apart — Hard
https://leetcode.com/problems/rearrange-string-k-distance-apart/
Best: 大根堆 + 冷却队列贪心 — O(n log 26)
"""

from collections import Counter, deque
import heapq


class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        if k <= 1:
            return s
        heap = [(-c, ch) for ch, c in Counter(s).items()]
        heapq.heapify(heap)
        wait = deque()  # (ready_idx, -cnt, ch)
        res = []
        i = 0
        while heap or wait:
            while wait and wait[0][0] <= i:
                _, cnt, ch = wait.popleft()
                heapq.heappush(heap, (cnt, ch))
            if not heap:
                return ""
            cnt, ch = heapq.heappop(heap)
            res.append(ch)
            cnt += 1  # cnt is negative
            if cnt != 0:
                wait.append((i + k, cnt, ch))
            i += 1
        return "".join(res)


def _test():
    s = Solution()
    r = s.rearrangeString("aabbcc", 3)
    assert len(r) == 6 and r != ""
    # same letters at least k apart
    def ok(t, k):
        last = {}
        for i, c in enumerate(t):
            if c in last and i - last[c] < k:
                return False
            last[c] = i
        return True

    assert ok(r, 3)
    assert s.rearrangeString("aaabc", 3) == ""
    r2 = s.rearrangeString("aaadbbcc", 2)
    assert ok(r2, 2) and sorted(r2) == sorted("aaadbbcc")
    assert s.rearrangeString("a", 0) == "a"
    print("LC 358 OK")


if __name__ == "__main__":
    _test()
