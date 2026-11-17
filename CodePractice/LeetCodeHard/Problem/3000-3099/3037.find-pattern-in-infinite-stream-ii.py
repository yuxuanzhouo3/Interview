"""
LC 3037. Find Pattern in Infinite Stream II — Hard
Best: KMP on stream — O(n+m)
"""

# InfiniteStream mock
class InfiniteStream:
    def __init__(self, data):
        self.data = data
        self.i = 0
    def next(self):
        v = self.data[self.i]
        self.i += 1
        return v


class Solution:
    def findPattern(self, stream: "InfiniteStream", pattern: list) -> int:
        m = len(pattern)
        # KMP LPS
        lps = [0] * m
        length = 0
        i = 1
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        i = j = 0  # i index in stream, j in pattern
        while True:
            x = stream.next()
            while j and x != pattern[j]:
                j = lps[j - 1]
            if x == pattern[j]:
                j += 1
                if j == m:
                    return i - m + 1
            i += 1


def _test():
    s = Solution()
    assert s.findPattern(InfiniteStream([1, 2, 3, 4, 5, 6, 7]), [3, 4]) == 2
    print("LC 3037 OK")


if __name__ == "__main__":
    _test()
