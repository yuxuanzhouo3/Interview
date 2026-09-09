import sys, collections
from collections import deque
from functools import lru_cache

N = -1
K = 0
A = []
for line in sys.stdin:
    a = line.split()
    if N == -1:
        N = int(a[0])
    elif K < N:
        A.append(list(a[0]))
        K +=1

# print(N, A)


D = [(-1, 0), (0, 1), (1, 0), (0, -1)]
distance = {} # 记忆剪枝
# house_to_cells = collections.defaultdict(int) # each house to all cell(x, y) distance accumlation
dist_sum = [[0] * N for _ in range(N)] # = 这个点到所有 house 的距离总和

# @lru_cache(maxsize=None)
def bfs2(house):
    # O(H * V) 
    for hx, hy in house:
        q = deque([(hx, hy)])
        step = -1
        V = [[0] * N for _ in range(N)]
        V[hx][hy] = 1
        while q:
            n = len(q)
            step += 1
            for _ in range(n):
                x, y = q.popleft()
                V[x][y] = 1
                dist_sum[x][y] += step
                for d in D:
                    nx, ny = x+d[0], y+d[1]
                    if 0<=nx<N and 0<=ny<N and A[nx][ny] != "*" and V[nx][ny]==0:
                        V[nx][ny] = 1
                        q.append((nx, ny))


def bfs(i, j, v):
    q = deque([(i, j)])
    house = []
    if A[i][j] == "#":
        house.append((i, j))
    mp = [(i, j)]
    # print(mp, house)
    while q:
        n = len(q)
        # print(q, v, n)
        for _ in range(n):
            x, y = q.popleft()
            v[x][y] = 1
            for d in D:
                nx, ny = x+d[0], y+d[1]
                # if (x, y) == (4 ,1) : print("UUUU", x, y, nx, ny, d, d[0], d[1])
                if 0<=nx<N and 0<=ny<N and A[nx][ny] != "*" and v[nx][ny]==0:
                     v[nx][ny] = 1
                     q.append((nx, ny))
                     mp.append((nx, ny))
                     if A[nx][ny] == "#":  # find house
                        house.append((nx, ny))
                

    d = float("inf")    
    if not house: return d 
    # print( sorted(house), sorted(mp))

    # O(V * H * V) -> V is numbers of cells <= N*N; H is numbers of houses <= N*N
    '''
    for mx, my in mp:
        cur = 0
        for hx, hy in house:
            # cur += abs(hx-mx) + abs(hy - my)
            if (mx, my, hx, hy) in distance:
                cur += distance[(mx, my, hx, hy)]
            else:
                cur +=  bfs2(mx, my, hx, hy)
        d = min(d, cur)
        # print(d, cur, mx, my)
    return d
    '''

    # O(V * H + V)
    
    bfs2(house) #  O(H * V) -> for all hx,hy; update dist_sum[mx][my]
    
    d = float("inf")
    for mx, my in mp: # O(V)
        d = min(d, dist_sum[mx][my])
    # print(d, cur, mx, my)
    return d


v = [[0] * N for _ in range(N)]
ans = 0
dis = 0

# O(V)
for i in range(N):
    for j in range(N):
        if A[i][j] != '*' and v[i][j] == 0:
            # print(v, i, j)
            d = bfs(i, j, v) 
            if d != float('inf'):
                ans += 1
                dis += d 
            
print(ans, dis)
