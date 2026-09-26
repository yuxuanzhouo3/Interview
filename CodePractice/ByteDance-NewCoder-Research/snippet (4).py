from functools import lru_cache
import sys, heapq

'''
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
'''

input = sys.stdin.readline
N = int(input())
# E = []
# G = [[] for _ in range(N)]
M = [[0] * N for _ in range(N)]

for i in range(N):
    M[i] = list(map(int, input().split()))
    # A = list(map(int, input().split()))
    # for j in range(N):
        # if i != j: E.append((i, j, A[j]))
        # G[i].append((j, A[j]))
        # print(i, j, M)
    #   M[i][j] = A[j]
        # print(M)

'''
def dfs(i, v, a, s):
    if a >= ans[0]:
        return

    if len(v) == N:
        ans[0] = min(ans[0], a+M[s][i])
        return 
    
    for j, w in enumerate(M[i]):
        if j !=i and j not in v:
            v.add(j)
            dfs(j, v, a+w, s)
            v.remove(j)
   
ans = [float('inf')]
# for i in range(N): dfs(i, set([i]), 0, i)
dfs(0, set([0]), 0, 0)
'''
'''
@lru_cache(maxsize=None)
def dfs2(i, mask, a, s):
    if a >= ans2[0]:
        return

    if mask == 0:
        # print(i)
        ans2[0] = min(ans2[0], a+M[s][i])
        return 
    
    for j, w in enumerate(M[i]):
        if j !=i and (1<<j) & mask :
            dfs2(j, mask ^ (1<<j), a+w, s)
   
ans2 = [float('inf')]
mask = (1 << N) - 1
# print(bin(mask), bin(mask ^ 1), bin(mask ^ (1<<2)))
# for i in range(N): dfs(i, set([i]), 0, i)
dfs2(0, mask ^ 1, 0, 0)
# print(ans2[0])
'''

@lru_cache(maxsize=None)
def dfs3(i, mask):
    if mask == 0:
        return M[i][0]
    ans = float('inf')
    for j, w in enumerate(M[i]):
        if (1 << j) & mask:
            ans = min( ans, w + dfs3( j, mask ^ (1 << j) ))
    return ans




@lru_cache(None)
def dfs4(i, mask):
    if mask == 0:
        return M[i][0]
    ans = 10**9
    x = mask
    while x:
        bit = x & -x
        j = bit.bit_length() - 1
        cost = M[i][j] + dfs4(j, mask ^ bit)
        if cost < ans:
            ans = cost
        x ^= bit
    return ans




@lru_cache(None)
def dfs5(i, mask):

    if mask == 0:
        return M[i][0]

    ans = 10**9
    row = M[i]

    x = mask

    while x:
        bit = x & -x
        j = BIT_TO_CITY[bit]

        v = row[j] + dfs5(
            j,
            mask ^ bit
        )

        if v < ans:
            ans = v

        x -= bit

    return ans



# mask 只表示城市 1...N-1
K = N - 1
FULL = (1 << K) - 1

# bit -> 实际城市编号
BIT_TO_CITY = {
    1 << i: i + 1
    for i in range(K)
}

mask = (1 << N) - 1
import time
t1 = time.perf_counter()
print( dfs5( 0, FULL ) )
t2 = time.perf_counter()
print(f"Solution: {t2-t1:.6f}s")


'''
class DSU:

    def __init__(self, n):

        self.parent = list(range(n))

        self.size = [1] * n


    def find(self, x):

        if self.parent[x] != x:

            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]


    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        self.size[ra] += self.size[rb]

        return True


    def connected(self, a, b):

        return self.find(a) == self.find(b)

def kruskal(n, edges):
    edges = sorted( edges, key=lambda x: x[2] )
    dsu = DSU(n)
    total = 0
    used = 0
    mst = []
    for u, v, w in edges:
        if dsu.union(u, v):
            total += w
            used += 1
            mst.append((u, v, w))
            if used == n - 1:
                break
    if used != n - 1:
        return None
    return total, mst

def prim(graph, start=0):
    n = len(graph)
    visited = [False] * n
    pq = [(0, start, -1)]
    total = 0
    used = 0
    mst = []

    while pq:
        w, u, parent = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        used += 1
        if parent != -1:
            mst.append( ( parent, u, w ) )

        for v, cost in graph[u]:
            if not visited[v]:
                heapq.heappush( pq, (cost, v, u))
    if used != n:
        return None
    return total, mst

input = sys.stdin.readline
N = int(input())
E = []
G = [[] for _ in range(N)]

for i in range(N):
    A = list(map(int, input().split()))
    for j in range(N):
        if i != j:
            E.append((i, j, A[j]))
        G[i].append((j, A[j]))

def connect_start_end(mst, N):
    seen = [0] * N
    for u, v, _ in mst:
        seen[u] += 1
        seen[v] += 1
    
    res = []
    for k, v  in enumerate(seen):
        if v == 1:
            res.append(k)
            break
    return sorted(res)


t, mst = kruskal(N, E)
t2, mst2 = prim(G)

print(E, t, mst)
# print(G, t2, mst2)

a, b = connect_start_end(mst, N)
print(a, b)

for u, v, w in E:
    if u == a and v == b:
        t += w
        break

for i in range(N):
    for j, w in G[i]:
        if i == a and j == b:
            t2 += w
            break

# print(E, t, mst)
# print(G, t2, mst2)


print(t)
'''