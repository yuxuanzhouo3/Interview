import collections


class DSU(object):
    def __init__(self, N):
        self.parent = list(range(N))
        # self.count = [0] * N
        self.count = N

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def find_half_path_compression(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if py > px:
            px, py = py, px
        self.parent[py] = px
        self.count -= 1


  # "\ 0 /",
  #  1 . 3
  # "/ 2 \"
def index(m, n, k, C):
    return (m * C + n) * 4 + k

def regionsBySlashes_dsu(A):
    if not A or not A[0]: return -1
    M, N = len(A), len(A[0])
    dsu = DSU(M * N * 4)
    for i in range(M):
        for j in range(N):
            root = (i * N + j) * 4

            if A[i][j] == '/':
                dsu.union(root, root+1)
                dsu.union(root+2, root+3)
            elif A[i][j] == ' ':
                dsu.union(root, root + 1)
                dsu.union(root + 1, root + 2)
                dsu.union(root + 2, root + 3)
            else:
                dsu.union(root, root + 3)
                dsu.union(root + 1, root + 2)

            if j + 1 < N:
                dsu.union(root+3, root+4+1)
                # dsu.union(root + 3, index(i, j + 1, 1, N))

            if i + 1 < M:
                dsu.union(root+2, root+4*N)
                # dsu.union(root+2, index(i+1, j, 0, N))

    # print(dsu.parent, dsu.count)
    return dsu.count


def regionsBySlashes_dfs(A):
    if not A or not A[0]: return -1
    M, N = len(A), len(A[0])
    graph = collections.defaultdict(set)
    for i in range(M):
        for j in range(N):
            root = (i * N + j) * 4
            if A[i][j] == '/':
                graph[root].add(root+1)
                graph[root+1].add(root)
                graph[root+2].add(root+3)
                graph[root+3].add(root+2)
            elif A[i][j] == '\\':
                graph[root].add(root + 3)
                graph[root + 3].add(root)
                graph[root + 2].add(root + 1)
                graph[root + 1].add(root + 2)
            else:
                graph[root].add(root + 1)
                graph[root].add(root + 2)
                graph[root].add(root + 3)
                graph[root + 1].add(root)
                graph[root + 1].add(root+2)
                graph[root + 1].add(root+3)
                graph[root + 2].add(root)
                graph[root + 2].add(root + 1)
                graph[root + 2].add(root + 3)
                graph[root + 3].add(root)
                graph[root + 3].add(root + 1)
                graph[root + 3].add(root + 2)

            if i + 1 < M:
                graph[root + 2].add(root + 4 * N)
                graph[root + 4 * N].add(root + 2)

            if j + 1 < N:
                graph[root + 3].add(root + 4 + 1)
                graph[root + 4 + 1].add(root + 3)

    v = set()
    ans = [0]
    def dfs(root):
        if root not in graph:
            return
        # ans[0] += 1
        for nei in graph[root]:
            if nei not in v:
                v.add(nei)
                dfs(nei)

    for i in range(M):
        for j in range(N):
            for k in range(4):
                if index(i, j, k, N) not in v:
                    ans[0] += 1
                    v.add(index(i, j, k, N))
                    dfs(index(i, j, k, N))

    return ans[0]

def regionsBySlashes_bfs(A):
    if not A or not A[0]: return -1
    M, N = len(A), len(A[0])
    graph = collections.defaultdict(set)
    for i in range(M):
        for j in range(N):
            root = (i * N + j) * 4
            if A[i][j] == '/':
                graph[root].add(root + 1)
                graph[root + 1].add(root)
                graph[root + 2].add(root + 3)
                graph[root + 3].add(root + 2)
            elif A[i][j] == '\\':
                graph[root].add(root + 3)
                graph[root + 3].add(root)
                graph[root + 2].add(root + 1)
                graph[root + 1].add(root + 2)
            else:
                graph[root].add(root + 1)
                graph[root].add(root + 2)
                graph[root].add(root + 3)
                graph[root + 1].add(root)
                graph[root + 1].add(root + 2)
                graph[root + 1].add(root + 3)
                graph[root + 2].add(root)
                graph[root + 2].add(root + 1)
                graph[root + 2].add(root + 3)
                graph[root + 3].add(root)
                graph[root + 3].add(root + 1)
                graph[root + 3].add(root + 2)

            if i + 1 < M:
                graph[root + 2].add(root + 4 * N)
                graph[root + 4 * N].add(root + 2)

            if j + 1 < N:
                graph[root + 3].add(root + 4 + 1)
                graph[root + 4 + 1].add(root + 3)

    v = set()
    ans = [0]

    def bfs(root):
        if root not in graph:
            return
        # ans[0] += 1
        q = [root]
        while q:
            node = q.pop(0)
            for nei in graph[node]:
                if nei not in v:
                    v.add(nei)
                    q.append(nei)

    for i in range(M):
        for j in range(N):
            for k in range(4):
                if index(i, j, k, N) not in v:
                    ans[0] += 1
                    v.add(index(i, j, k, N))
                    bfs(index(i, j, k, N))

    return ans[0]

# class DSU:
#     def __init__(self, n):
#         self.parent = list(range(n))
#         self.count = n  # 连通块数量
#
#     def find(self, x):
#         while self.parent[x] != x:
#             self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
#             x = self.parent[x]
#         return x
#
#     def union(self, x, y):
#         rx, ry = self.find(x), self.find(y)
#         if rx != ry:
#             self.parent[rx] = ry
#             self.count -= 1

def regionsBySlashes(grid):
    n = len(grid)
    dsu = DSU(n * n * 4)

    def idx(r, c, k):
        return (r * n + c) * 4 + k

    for r in range(n):
        for c in range(n):
            root = (r * n + c) * 4
            val = grid[r][c]

            # 同一个格子内部连通情况
            if val == ' ':
                dsu.union(root + 0, root + 1)
                dsu.union(root + 1, root + 2)
                dsu.union(root + 2, root + 3)
            elif val == '/':
                dsu.union(root + 0, root + 3)
                dsu.union(root + 1, root + 2)
            else:  # '\\'
                dsu.union(root + 0, root + 1)
                dsu.union(root + 2, root + 3)

            # 与右边格子的连接
            if c + 1 < n:
                dsu.union(root + 1, idx(r, c + 1, 3))
            # 与下边格子的连接
            if r + 1 < n:
                dsu.union(root + 2, idx(r + 1, c, 0))

    # print(dsu.parent, dsu.count)
    return dsu.count

def regionsBySlashesDFS(grid):
    n = len(grid)
    size = n * n * 4
    graph = [[] for _ in range(size)]

    def idx(r, c, k):
        return (r * n + c) * 4 + k

    for r in range(n):
        for c in range(n):
            root = (r * n + c) * 4
            val = grid[r][c]

            # 内部连接
            if val == ' ':
                for i in range(4):
                    for j in range(i + 1, 4):
                        graph[root + i].append(root + j)
                        graph[root + j].append(root + i)
            elif val == '/':
                graph[root + 0].append(root + 3)
                graph[root + 3].append(root + 0)
                graph[root + 1].append(root + 2)
                graph[root + 2].append(root + 1)
            else:  # '\\'
                graph[root + 0].append(root + 1)
                graph[root + 1].append(root + 0)
                graph[root + 2].append(root + 3)
                graph[root + 3].append(root + 2)

            # 邻居连接
            if c + 1 < n:
                graph[root + 1].append(idx(r, c + 1, 3))
                graph[idx(r, c + 1, 3)].append(root + 1)
            if r + 1 < n:
                graph[root + 2].append(idx(r + 1, c, 0))
                graph[idx(r + 1, c, 0)].append(root + 2)

    visited = [False] * size

    def dfs(u):
        stack = [u]
        visited[u] = True
        while stack:
            node = stack.pop()
            for nxt in graph[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)

    count = 0
    for i in range(size):
        if not visited[i]:
            dfs(i)
            count += 1
    return count

from collections import deque
def regionsBySlashesBFS(grid):
    n = len(grid)
    size = n * n * 4
    graph = [[] for _ in range(size)]

    def idx(r, c, k):
        return (r * n + c) * 4 + k

    for r in range(n):
        for c in range(n):
            root = (r * n + c) * 4
            val = grid[r][c]

            # 内部连接
            if val == ' ':
                for i in range(4):
                    for j in range(i + 1, 4):
                        graph[root + i].append(root + j)
                        graph[root + j].append(root + i)
            elif val == '/':
                graph[root + 0].append(root + 3)
                graph[root + 3].append(root + 0)
                graph[root + 1].append(root + 2)
                graph[root + 2].append(root + 1)
            else:  # '\\'
                graph[root + 0].append(root + 1)
                graph[root + 1].append(root + 0)
                graph[root + 2].append(root + 3)
                graph[root + 3].append(root + 2)

            # 邻居连接
            if c + 1 < n:
                graph[root + 1].append(idx(r, c + 1, 3))
                graph[idx(r, c + 1, 3)].append(root + 1)
            if r + 1 < n:
                graph[root + 2].append(idx(r + 1, c, 0))
                graph[idx(r + 1, c, 0)].append(root + 2)

    visited = [False] * size
    count = 0

    for i in range(size):
        if not visited[i]:
            count += 1
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                for nxt in graph[node]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        queue.append(nxt)
    return count


# 测试示例
grid1 = [
  " /",
  "/ "
]

grid2 = [
  " /",
  "  "
]

print(regionsBySlashes(grid1))  # 输出：2
print(regionsBySlashesDFS(grid1))  # 输出：2
print(regionsBySlashesBFS(grid1))  # 输出：2

print(regionsBySlashes_dsu(grid1))  # 输出：2
print(regionsBySlashes_dfs(grid1))  # 输出：2
print(regionsBySlashes_bfs(grid1))  # 输出：2

# print(regionsBySlashes(grid2))  # 输出：1
# print(regionsBySlashesDFS(grid2))  # 输出：1
# print(regionsBySlashesBFS(grid2))  # 输出：1
#
# print(regionsBySlashes_dsu(grid2))  # 输出：1
# print(regionsBySlashes_dfs(grid2))  # 输出：1
# print(regionsBySlashes_bfs(grid2))  # 输出：1
