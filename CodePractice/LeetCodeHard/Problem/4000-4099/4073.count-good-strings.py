"""
LC 4073. Count Good Strings — Hard
https://leetcode.com/problems/count-good-strings/
Best: closed form / matrix expo for odd-run concatenations of a/b
"""
MOD = 10**9 + 7


class Solution:
    def countGoodStrings(self, n: int) -> int:
        # Good strings = concatenations of odd-length mono runs with alternating letters.
        # Equivalent: split n into ordered odd positive parts, 2 choices for starting letter.
        # Number of compositions of n into odd parts = 2^{c-1} where c = number of parts? 
        # Actually #odd compositions of n is Fibonacci: F_{(n+1)/2} style.
        # Let a_n = # good strings of length n.
        # From examples: n=1: "a","b" -> 2; n=2: ab,ba -> 2; n=3: 4; n=4: 6
        # Recurrence: a good string ends with an odd mono run of length 2k+1 of letter L,
        # previous (if any) ends with other letter and is good.
        # a_n = 2 * sum_{k>=0, 2k+1<=n} [1 if n==2k+1 else a_{n-(2k+1)} / something]
        # Simpler DP for small and matrix for large:
        # dp[i][0/1][parity of current run][...] 
        # Observation from definition: good iff can partition into odd mono blocks alternating.
        # So uniquely determined by starting letter and cut positions after odd lengths.
        # Number of ways to write n = o1+o2+...+ot with each oi odd >=1: equals 2^{t-1}? 
        # #compositions of n with odd parts = fib(n).
        # For each composition with t parts, 2 choices of starting letter (then alternating forced).
        # So a_n = 2 * (#odd compositions of n) = 2 * F_n where F is Fibonacci with F_1=1,F_2=1,F_3=2...
        # n=1: compositions (1) -> 1, a=2. OK
        # n=2: (1+1) only -> 1, a=2. OK
        # n=3: (1+1+1),(3) -> 2, a=4. OK
        # n=4: (1+1+1+1),(1+3),(3+1) -> 3, a=6. OK
        # Yes! a_n = 2 * (# odd-part compositions of n).
        # And # odd-part compositions of n equals 2^{n-1} for? No - fibonacci:
        # Let f(n) = #odd compositions. f(n)=f(n-1)+f(n-2) with f(0)=1,f(1)=1.
        # Standard: f(n) = fib(n) with fib(1)=1,fib(2)=1,fib(3)=2,fib(4)=3,fib(5)=5.
        # Wait f(2)=1 (1+1), fib(2)=1; f(3)=2, fib(3)=2; f(4)=3. Yes f(n)=fib(n).
        # With fib(1)=1,fib(2)=1.
        # a_n = 2 * fib(n) mod 1e9+7, n up to 1e15 -> matrix expo.

        def mat_mul(A, B):
            return [
                [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD, (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD],
                [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD, (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD],
            ]

        def mat_pow(M, e):
            R = [[1, 0], [0, 1]]
            while e:
                if e & 1:
                    R = mat_mul(R, M)
                M = mat_mul(M, M)
                e >>= 1
            return R

        if n == 0:
            return 1
        # fib(n): [[1,1],[1,0]]^(n-1) * [1,1]^T -> fib(n), fib(n-1)
        M = mat_pow([[1, 1], [1, 0]], n - 1)
        fib_n = M[0][0]  # since [[1,1],[1,0]]^(n-1) * [fib2,fib1]=[fib(n+1)?,...]
        # With F1=1,F2=1: M^(n-1) applied to [F2,F1]=[1,1] gives [F_{n+1}, F_n]? 
        # [[1,1],[1,0]] * [F_k, F_{k-1}] = [F_k+F_{k-1}, F_k] = [F_{k+1}, F_k]
        # Start [F2,F1]=[1,1], after n-1: [F_{n+1}, F_n]
        # So fib_n = M[1][0]*1 + M[1][1]*1 = M[1][0]+M[1][1]
        # Or M[0][1] + M[1][1] ...
        fib_n = (M[0][1] + M[1][1]) % MOD  # F_n when n>=1: after n-1 steps from [1,1] second component
        # Verify: n=1 -> M^0=I, fib= I[0][1]+I[1][1]=0+1=1 OK
        # n=2 -> M=[[1,1],[1,0]], fib=1+0=1 OK
        # n=3 -> M^2=[[2,1],[1,1]], fib=1+1=2 OK
        # n=4 -> M^3=[[3,2],[2,1]], fib=2+1=3 OK
        return (2 * fib_n) % MOD


def _test():
    s = Solution()
    assert s.countGoodStrings(4) == 6
    assert s.countGoodStrings(3) == 4
    assert s.countGoodStrings(2) == 2
    assert s.countGoodStrings(1) == 2
    print("LC 4073 OK")


if __name__ == "__main__":
    _test()
