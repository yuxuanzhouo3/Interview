"""
LC 2014. Longest Subsequence Repeated k Times — Hard
https://leetcode.com/problems/longest-subsequence-repeated-k-times/
Best: BFS longest candidate subsequence check — O(|Σ|^L * n)
"""


from collections import Counter, deque


class Solution:
    def longestSubsequenceRepeatedK(self, s: str, k: int) -> str:
        cnt = Counter(s)
        chars = sorted([c for c in cnt if cnt[c] >= k], reverse=True)

        def ok(t: str) -> bool:
            # t * k is subsequence of s
            i = 0
            need = t * k
            for ch in s:
                if ch == need[i]:
                    i += 1
                    if i == len(need):
                        return True
            return False

        best = ""
        q = deque([""])
        while q:
            cur = q.popleft()
            for c in chars:
                nxt = cur + c
                if ok(nxt):
                    if len(nxt) > len(best) or (len(nxt) == len(best) and nxt > best):
                        best = nxt
                    q.append(nxt)
        return best


def _test():
    s = Solution()
    assert s.longestSubsequenceRepeatedK("letsleetcode", 2) == "let"
    assert s.longestSubsequenceRepeatedK("bb", 2) == "b"
    assert s.longestSubsequenceRepeatedK("ab", 2) == ""
    print("LC 2014 OK")


if __name__ == "__main__":
    _test()

