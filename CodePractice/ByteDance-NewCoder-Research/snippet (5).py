import sys

input = sys.stdin.readline
N = int(input())
H = list(map(int, input().split()))

def can_win(e):
    for _, h in enumerate(H):
        if h > e:
            e -= (h-e)
        else:
            e += (e-h)
        if e <= 0:
            return False
    return True

l, r = 1, 10**5 + 1

while l < r:
    mid = l + (r - l) // 2

    if can_win(mid):
        r = mid 
    else:
        l = mid + 1

print(l)



