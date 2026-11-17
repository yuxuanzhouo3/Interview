"""
LC 4060. Count Evenly Good Integers — Hard
https://leetcode.com/problems/count-evenly-good-integers/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O(logr)
# Space: O(1)

# math
class Solution(object):
    def countEvenlyGoodIntegers(self, l, r):
        """
        :type l: int
        :type r: int
        :rtype: int
        """
        def count(n):
            if n <= 0:
                return 0
            q, parity = n//10, 0
            while q:
                q, d = divmod(q, 10)
                parity ^= d%2 == 0
            q, r = divmod(n, 10)
            return q*5+r//2+1 if parity else q*5+(r+1)//2

        return count(r)-count(l-1)

def _test():
    s = Solution()
    assert s.countEvenlyGoodIntegers(l=18, r=22) == 3
    assert s.countEvenlyGoodIntegers(l=98, r=101) == 2
    assert s.countEvenlyGoodIntegers(l=1, r=10) == 5
    print("LC 4060 OK")


if __name__ == "__main__":
    _test()
