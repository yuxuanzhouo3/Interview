import sys

input = sys.stdin.readline

n, m = map(int, input().split())
A = list(map(int, input().split()))

if 3 * m < n:
    print(-1)
    sys.exit()

m = min(m, n)

if m == n:
    print(sum(A))
    sys.exit()

NEG = -10**9

prev = [NEG] * (n + 1)

# 第 1 步
remain = m - 1

left = max(1, n - 3 * remain)
right = min(3, n - remain)

for pos in range(left, right + 1):
    prev[pos] = A[pos - 1]

# 普通 Python for 2-d DP            O(N * M) * a; O(N*M) space
# 普通 Python for rolling DP      ⚠️ 约 4 秒 O(N * M) * a; O(N) space
# list-comprehension rolling DP   ✅ 更适合现在这个卡常数情况 O(N * M) * b; O(N) space; b < a
# 普通 Python for       50,000,000 × 较大的常数
# list comprehension    50,000,000 × 较小的常数

for step in range(2, m + 1):

    remain = m - step

    left = max(
        step,
        n - 3 * remain
    )

    right = min(
        3 * step,
        n - remain
    )

    cur = [NEG] * (n + 1)

    start = left

    # 特殊处理 pos = 2
    if start == 2:
        cur[2] = A[1] + max(
            prev[1],
            prev[0]
        )
        start = 3

    # 主区间：pos >= 3
    if start <= right:

        cur[start:right + 1] = [
            a + max(x, y, z)

            for a, x, y, z in zip(
                A[start - 1:right],
                prev[start - 1:right],
                prev[start - 2:right - 1],
                prev[start - 3:right - 2]
            )
        ]

    prev = cur


ans = prev[n]

print(ans if ans != NEG else -1)