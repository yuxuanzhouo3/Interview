import sys

n = m = -1
A = []
k = -1
B = []
nb = 0
for line in sys.stdin:
    a = line.split()
    if n == -1 and m == -1:
        n, m = int(a[0]), int(a[1])
    elif not A:
        A = list(map(int, a))
    elif k == -1:
        k = int(a[0])
    elif nb < k:
        B.append(list(map(int, a)))
        nb += 1
    else:
        print("error")
    

# print(m, n , A, k, B )

A_ = []
n_ = n - k

for i, j in B:
    A_.append((A[i-1] + A[j-1], 2))
    A[i-1] = A[j-1] = 0

for i in range(n):
    if A[i]:
        A_.append((A[i], 1))

A_.sort()


dp = [ [0] * (m+1) for _ in range(n_) ]
# dp[i][v]: 0-ith items total <= v -> max item numbers
candy, ox_num = A_[0]
if m<candy: print(0)
for v in range(1, m+1):
    dp[0][v] = ox_num
for i in range(n_):
    V, K = A_[i]
    for v in range(m, -1, -1): # backward 0-1 knapsack; forward unbounded knapsack;
        if v >= V:
            # dp[i][v] = max(dp[i][v], dp[i-1][v])
            dp[i][v] = max(dp[i-1][v], dp[i-1][v-V] + K)
        else:
            dp[i][v] = dp[i-1][v]

print(dp[n_-1][m])


'''

‘’‘‘’‘
dp = [0] * (m + 1)

for i in range(n_):
    V, K = A_[i]
    for v in range(m, V-1, -1):
        dp[v] = max(dp[v], dp[v-V] + K)

print(dp[m])
’‘’’‘’

# 0-1-knap-pack
[1, 2, 2, 1, 1, 2, ... ]
[1, 3, 4, 5, 6, 10, ...]

[2, 2, 2, 1, 1, 1, ....]
[3, 4, 10,1, 5, 6, ....]

# greedy --> error ??
[1, 2, 1, 2, 1, 2,  1, ....]
[1, 3, 2, 4, 5, 10, 6, ....]
'''


