import sys

input = sys.stdin.readline

M, N = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(M)]

INF = 10**18
dp = [[INF] * (N + 1) for _ in range(M + 1)]

# 终点
dp[M - 1][N - 1] = max(
    1,
    1 - A[M - 1][N - 1]
)

# 右下 -> 左上
for i in range(M - 1, -1, -1):
    for j in range(N - 1, -1, -1):

        if i == M - 1 and j == N - 1:
            continue

        need = min(
            dp[i + 1][j],
            dp[i][j + 1]
        )

        dp[i][j] = max(
            1,
            need - A[i][j]
        )

print(dp[0][0])


'''
d = [(0, 1), (1, 0)]


def can_reach(V):

    # 初始化到左上角
    V += A[0][0]

    if V <= 0:
        return False

    # best[x][y]：
    # 到达这个格子时，最大的剩余行动力
    best = [[-1] * N for _ in range(M)]
    best[0][0] = V

    q = collections.deque([(0, 0)])

    while q:

        x, y = q.popleft()
        V = best[x][y]

        if x == M - 1 and y == N - 1:
            return True

        for dx, dy in d:

            nx = x + dx
            ny = y + dy

            if 0 <= nx < M and 0 <= ny < N:

                nv = V + A[nx][ny]

                if nv > 0 and nv > best[nx][ny]:
                    best[nx][ny] = nv
                    q.append((nx, ny))

    return False


l, r = 1, 200000

while l < r:

    mid = (l + r) // 2

    if can_reach(mid):
        r = mid
    else:
        l = mid + 1

print(l)


# print(M, N, A)



dp = [[0] * N for _ in range(M)]
dp[-1][-1] = 1 # [1, 200000]
ab = [0]

for i in range(M-2, -1, -1):
    dp[i][-1] = dp[i+1][-1] - A[i][-1]

for j in range(N-2, -1, -1):
    dp[-1][j] = dp[-1][j+1] - A[-1][j]

for i in range(M-2, -1, -1):
    for j in range(N-2, -1, -1):
        dp[i][j] = max(dp[i+1][j] - A[i][j], dp[i][j+1] - A[i][j])
        # if dp[i][j] <= 0: ab.append(dp[i][j])


mn = 1e8
for i in range(M):
    for j in range(N):
        mn = min(dp[i][j], mn)

if mn < 0:
    print( -mn )
else:
    print(dp[0][0])



'''


