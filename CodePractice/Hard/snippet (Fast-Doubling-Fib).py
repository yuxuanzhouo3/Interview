# -*- coding:utf-8 -*-
import math 

class GoUpstairs:
    def countWays(self, n):
        # write code here
        M = 1000000007
        '''
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1
        '''
        '''
        pre1 = 1
        pre2 = 1
        cur = 0

        for i in range(2, n+1):
            # dp[i] = (dp[i-1] + dp[i-2]) % M 
            cur =  ( pre1 +  pre2 ) % M 
            pre1 = pre2
            pre2 = cur
        
        return max(pre2, cur)
        
        a, b = 0, 1
        for _ in range(n):
            a, b = b % M, (a + b) % M
        return a

# Fibonacci Fast Doubling

F(0) = 0
F(1) = 1

F(n + 2) = F(n + 1) + F(n)


# 1. Key identity

F(m + n) = F(m - 1) * F(n) + F(m) * F(n + 1)


# 2. Derive F(2k)

Set m = k, n = k:

F(2k)
= F(k - 1) * F(k) + F(k) * F(k + 1)

= F(k) * (F(k - 1) + F(k + 1))

Since:

F(k + 1) = F(k) + F(k - 1)

therefore:

F(k - 1) = F(k + 1) - F(k)

So:

F(2k)
= F(k) * (F(k + 1) - F(k) + F(k + 1))

= F(k) * (2 * F(k + 1) - F(k))

Therefore:

F(2k) = F(k) * (2 * F(k + 1) - F(k))


# 3. Derive F(2k + 1)

Set m = k + 1, n = k:

F(2k + 1)
= F(k) * F(k) + F(k + 1) * F(k + 1)

Therefore:

F(2k + 1)
= F(k)^2 + F(k + 1)^2


# 4. Fast Doubling formulas

F(2k)   = F(k) * (2 * F(k + 1) - F(k))

F(2k+1) = F(k)^2 + F(k + 1)^2


# 5. Python implementation

class Fibonacci:
    def fib(self, n):
        if n == 0:
            return 0, 1

        # k = n // 2
        # a = F(k)
        # b = F(k + 1)
        a, b = self.fib(n // 2)

        # F(2k)
        c = a * (2 * b - a)

        # F(2k + 1)
        d = a * a + b * b

        if n % 2 == 0:
            # n = 2k
            return c, d
        else:
            # n = 2k + 1
            return d, c + d


# 6. Example: calculate F(10)

# n = 10
#
# k = 5
#
# F(5)  = 5
# F(6)  = 8
#
# F(10)
# = F(5) * (2 * F(6) - F(5))
# = 5 * (2 * 8 - 5)
# = 5 * 11
# = 55
#
# F(11)
# = F(5)^2 + F(6)^2
# = 5^2 + 8^2
# = 25 + 64
# = 89
#
# return:
# F(10) = 55
# F(11) = 89


# 7. Why O(log n)?

# Normal Fibonacci:
#
# F(0)
# F(1)
# F(2)
# ...
# F(n)
#
# Time: O(n)
#
#
# Fast Doubling:
#
# n
# ↓
# n / 2
# ↓
# n / 4
# ↓
# n / 8
# ↓
# ...
# ↓
# 0
#
# Time: O(log n)
#
# Space: O(log n) because of recursion

        
        '''

        def fib(n):
            if n == 0:
                return 0, 1

            a, b = fib(n // 2)

            c = a * ((2 * b - a) % M) % M
            d = (a * a + b * b) % M

            if n & 1:
                return d, (c + d) % M
            else:
                return c, d

        return fib(n + 1)[0]