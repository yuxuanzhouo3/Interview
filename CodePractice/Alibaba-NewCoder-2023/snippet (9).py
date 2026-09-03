import sys

input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

pos = {x: i for i, x in enumerate(B)}

mx = cur = 1

for i in range(1, N):
    if pos[A[i]] > pos[A[i - 1]]:
        cur += 1
    else:
        cur = 1

    mx = max(mx, cur)

print(N - mx)