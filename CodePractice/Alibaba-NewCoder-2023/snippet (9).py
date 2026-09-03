import sys

input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

indexB = {x: i for i, x in enumerate(B)}

mx = cur = 1

# 所有从头到尾一次都没有被取出的元素，在原 A 中必须是连续的一整段。
# a b [c d e] f g; [c d e] 0 moves; b c and e f is not subarray;  a b f g at least move once

for i in range(1, N):
    if indexB[A[i]] > indexB[A[i - 1]]:
        cur += 1
    else:
        cur = 1

    mx = max(mx, cur)

print(N - mx) 