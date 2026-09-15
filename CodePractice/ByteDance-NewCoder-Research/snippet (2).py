import sys
from collections import Counter

input = sys.stdin.readline
A = list(map(int, input().split()))
res = []

def dfs2(cnt):
    """
    判断剩余牌能否全部组成：
    刻子 xxx 或 顺子 x,x+1,x+2
    """
    # 找当前最小的还存在的牌
    x = -1
    for i in range(1, 10):
        if cnt[i] > 0:
            x = i
            break

    # 全部使用完
    if x == -1:
        return True

    # 1. 尝试刻子
    if cnt[x] >= 3:
        cnt[x] -= 3
        if dfs2(cnt):
            cnt[x] += 3
            return True
        cnt[x] += 3

    # 2. 尝试顺子 x,x+1,x+2
    if ( x <= 7 and cnt[x] > 0 and cnt[x + 1] > 0 and cnt[x + 2] > 0 ):
        cnt[x] -= 1
        cnt[x + 1] -= 1
        cnt[x + 2] -= 1
        if dfs2(cnt):
            cnt[x] += 1
            cnt[x + 1] += 1
            cnt[x + 2] += 1
            return True
        cnt[x] += 1
        cnt[x + 1] += 1
        cnt[x + 2] += 1
    return False

def dfs(B):
    cnt = Counter(B)
    # 枚举雀头
    for x in range(1, 10):
        if cnt[x] >= 2:
            cnt[x] -= 2
            if dfs2(cnt):
                cnt[x] += 2
                return True
            cnt[x] += 2
    return False

# 枚举摸到 1~9
original = Counter(A)
for x in range(1, 10):
    # 同一种牌最多4张
    if original[x] >= 4:
        continue
    B = A + [x]
    if dfs(B):
        res.append(x)

print(*res if res else 0)
