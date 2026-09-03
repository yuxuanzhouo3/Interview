import sys

input = sys.stdin.readline
MOD = 10000

T = int(input())

for _ in range(T):

    n, m = map(int, input().split())

    A = [
        list(map(int, input().split()))
        for _ in range(n)
    ]

    dp = [[0] * m for _ in range(n)]
    dp[0][0] = 1

    # DP -> O((m+n)^2 * m * n)  ==> Dig / Tree O(logm * logn * m * n)
    for i in range(n):
        for j in range(m):

            energy = A[i][j]

            for down in range(energy + 1):

                x = i + down

                if x >= n:
                    break

                max_right = energy - down

                for right in range(max_right + 1):

                    if down == 0 and right == 0:
                        continue

                    y = j + right

                    if y >= m:
                        break

                    dp[x][y] = (
                        dp[x][y] + dp[i][j]
                    ) % MOD

    print(dp[n - 1][m - 1])