"""
LC 1542. Find Longest Awesome Substring — Hard
https://leetcode.com/problems/find-longest-awesome-substring/
Best: Bitmask prefix parity + first occurrence — O(n · 10)
"""


class Solution:
    def longestAwesome(self, s: str) -> int:
        # mask of digit parity; palindrome after rearrange => at most one odd
        first = {0: -1}
        mask = 0
        ans = 0
        for i, ch in enumerate(s):
            mask ^= 1 << (ord(ch) - 48)
            if mask in first:
                ans = max(ans, i - first[mask])
            else:
                first[mask] = i
            for d in range(10):
                m2 = mask ^ (1 << d)
                if m2 in first:
                    ans = max(ans, i - first[m2])
        return ans


def _test():
    s = Solution()
    assert s.longestAwesome("3242415") == 5
    assert s.longestAwesome("213123") == 6
    assert s.longestAwesome("00") == 2
    print("LC 1542 OK")


if __name__ == "__main__":
    _test()
