'''
Graph Theory / Graph Algorithms
│
├── 1. Graph Representation（图的表示）
│   ├── Adjacency List          邻接表
│   ├── Adjacency Matrix        邻接矩阵
│   ├── Edge List               边集
│   └── Grid Graph              网格图 ★ 你的超市题
│
├── 2. Graph Traversal（图遍历）
│   ├── BFS
│   │   ├── Level-order traversal
│   │   ├── Unweighted Shortest Path ★
│   │   └── Multi-source BFS
│   │
│   └── DFS
│       ├── Recursive DFS
│       └── Iterative DFS
│
├── 3. Connectivity（连通性）
│   ├── Connected Components    连通分量 ★
│   │   ├── BFS
│   │   ├── DFS
│   │   └── Union Find
│   │
│   ├── Union Find / DSU
│   │   ├── find()
│   │   ├── union()
│   │   ├── Path Compression
│   │   └── Union by Rank/Size
│   │
│   ├── SCC                     强连通分量
│   │   ├── Tarjan
│   │   └── Kosaraju
│   │
│   ├── Bridge                  桥
│   └── Articulation Point      割点
│
├── 4. Shortest Path（最短路）
│   ├── BFS
│   │      Unweighted / equal-weight
│   │      O(V + E)
│   │
│   ├── 0-1 BFS
│   │      edge weight ∈ {0,1}
│   │      O(V + E)
│   │
│   ├── Dijkstra
│   │      weight >= 0
│   │      O((V+E) log V)
│   │
│   ├── Bellman-Ford
│   │      negative edges allowed
│   │      O(VE)
│   │
│   ├── Floyd-Warshall
│   │      All-Pairs Shortest Path
│   │      O(V³)
│   │
│   └── A*
│          start → target
│          heuristic search
│
├── 5. Minimum Spanning Tree（最小生成树）
│   ├── Kruskal
│   │   ├── Sort edges
│   │   └── Union Find ★
│   │
│   └── Prim
│       └── Priority Queue / Heap
│
├── 6. DAG（有向无环图）
│   ├── Topological Sort
│   │   ├── Kahn BFS
│   │   └── DFS
│   │
│   └── DAG DP
│
├── 7. Cycle Detection（环检测）
│   ├── Undirected
│   │   ├── DFS/BFS
│   │   └── Union Find
│   │
│   └── Directed
│       ├── DFS coloring
│       └── Topological Sort
│
├── 8. Network Flow（网络流）
│   ├── Max Flow
│   │   ├── Ford-Fulkerson
│   │   ├── Edmonds-Karp
│   │   └── Dinic
│   ├── Min Cut
│   └── Min-Cost Max-Flow
│
└── 9. Matching（匹配）
    ├── Bipartite Matching
    └── Hungarian / Hopcroft-Karp
'''


# ============================================================
# Graph Theory / Graph Algorithms - Python Template
# ============================================================

from collections import deque, defaultdict
import heapq


# ============================================================
# 1. GRAPH REPRESENTATION
# ============================================================

def build_undirected_graph(n, edges):
    """
    edges: [(u, v), ...]
    """
    graph = [[] for _ in range(n)]

    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    return graph


def build_directed_graph(n, edges):
    """
    edges: [(u, v), ...]
    """
    graph = [[] for _ in range(n)]

    for u, v in edges:
        graph[u].append(v)

    return graph


def build_weighted_graph(n, edges, undirected=True):
    """
    edges: [(u, v, w), ...]
    """
    graph = [[] for _ in range(n)]

    for u, v, w in edges:
        graph[u].append((v, w))

        if undirected:
            graph[v].append((u, w))

    return graph


# ============================================================
# 2. GRAPH TRAVERSAL
# ============================================================

# ------------------------------------------------------------
# 2.1 BFS
# Time: O(V + E)
# ------------------------------------------------------------

def bfs(start, graph):

    n = len(graph)

    visited = [False] * n
    q = deque([start])

    visited[start] = True

    order = []

    while q:

        u = q.popleft()

        order.append(u)

        for v in graph[u]:

            if not visited[v]:
                visited[v] = True
                q.append(v)

    return order


# ------------------------------------------------------------
# 2.2 BFS Level-order
# ------------------------------------------------------------

def bfs_level(start, graph):

    n = len(graph)

    visited = [False] * n
    visited[start] = True

    q = deque([start])

    level = 0
    ans = []

    while q:

        size = len(q)

        current_level = []

        for _ in range(size):

            u = q.popleft()

            current_level.append(u)

            for v in graph[u]:

                if not visited[v]:

                    visited[v] = True
                    q.append(v)

        ans.append(current_level)

        level += 1

    return ans


# ------------------------------------------------------------
# 2.3 DFS Recursive
# ------------------------------------------------------------

def dfs_recursive(start, graph):

    n = len(graph)

    visited = [False] * n

    order = []

    def dfs(u):

        visited[u] = True

        order.append(u)

        for v in graph[u]:

            if not visited[v]:
                dfs(v)

    dfs(start)

    return order


# ------------------------------------------------------------
# 2.4 DFS Iterative
# ------------------------------------------------------------

def dfs_iterative(start, graph):

    n = len(graph)

    visited = [False] * n

    stack = [start]

    order = []

    while stack:

        u = stack.pop()

        if visited[u]:
            continue

        visited[u] = True

        order.append(u)

        for v in reversed(graph[u]):

            if not visited[v]:
                stack.append(v)

    return order


# ============================================================
# 3. CONNECTIVITY
# ============================================================

# ------------------------------------------------------------
# 3.1 Connected Components - BFS
# ------------------------------------------------------------

def connected_components(graph):

    n = len(graph)

    visited = [False] * n

    components = []

    for start in range(n):

        if visited[start]:
            continue

        q = deque([start])

        visited[start] = True

        component = []

        while q:

            u = q.popleft()

            component.append(u)

            for v in graph[u]:

                if not visited[v]:

                    visited[v] = True
                    q.append(v)

        components.append(component)

    return components


# ------------------------------------------------------------
# 3.2 Union Find / DSU
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3.3 SCC - Tarjan
# ------------------------------------------------------------

def tarjan_scc(graph):

    n = len(graph)

    dfn = [-1] * n
    low = [0] * n

    stack = []

    in_stack = [False] * n

    timestamp = 0

    sccs = []

    def dfs(u):

        nonlocal timestamp

        dfn[u] = timestamp
        low[u] = timestamp

        timestamp += 1

        stack.append(u)

        in_stack[u] = True

        for v in graph[u]:

            if dfn[v] == -1:

                dfs(v)

                low[u] = min(
                    low[u],
                    low[v]
                )

            elif in_stack[v]:

                low[u] = min(
                    low[u],
                    dfn[v]
                )

        if low[u] == dfn[u]:

            component = []

            while True:

                x = stack.pop()

                in_stack[x] = False

                component.append(x)

                if x == u:
                    break

            sccs.append(component)

    for u in range(n):

        if dfn[u] == -1:
            dfs(u)

    return sccs


# ============================================================
# 4. SHORTEST PATH
# ============================================================

# ------------------------------------------------------------
# 4.1 BFS Shortest Path
# Edge weight = 1
# Time: O(V + E)
# ------------------------------------------------------------

def shortest_path_bfs(graph, start):

    n = len(graph)

    dist = [-1] * n

    dist[start] = 0

    q = deque([start])

    while q:

        u = q.popleft()

        for v in graph[u]:

            if dist[v] == -1:

                dist[v] = dist[u] + 1

                q.append(v)

    return dist


# ------------------------------------------------------------
# 4.2 0-1 BFS
# Edge weight in {0, 1}
# Time: O(V + E)
# graph[u] = [(v, weight), ...]
# ------------------------------------------------------------

def zero_one_bfs(graph, start):

    n = len(graph)

    INF = float("inf")

    dist = [INF] * n

    dist[start] = 0

    dq = deque([start])

    while dq:

        u = dq.popleft()

        for v, w in graph[u]:

            nd = dist[u] + w

            if nd < dist[v]:

                dist[v] = nd

                if w == 0:

                    dq.appendleft(v)

                else:

                    dq.append(v)

    return dist


# ------------------------------------------------------------
# 4.3 Dijkstra
# Edge weight >= 0
# Time: O((V + E) log V)
# ------------------------------------------------------------

def dijkstra(graph, start):

    n = len(graph)

    INF = float("inf")

    dist = [INF] * n

    dist[start] = 0

    pq = [(0, start)]

    while pq:

        d, u = heapq.heappop(pq)

        if d != dist[u]:
            continue

        for v, w in graph[u]:

            nd = d + w

            if nd < dist[v]:

                dist[v] = nd

                heapq.heappush(
                    pq,
                    (nd, v)
                )

    return dist


# ------------------------------------------------------------
# 4.4 Bellman-Ford
# Negative edge allowed
# Time: O(VE)
#
# edges = [(u, v, w), ...]
#
# return None => reachable negative cycle
# ------------------------------------------------------------

def bellman_ford(n, edges, start):

    INF = float("inf")

    dist = [INF] * n

    dist[start] = 0

    for _ in range(n - 1):

        changed = False

        for u, v, w in edges:

            if dist[u] == INF:
                continue

            if dist[u] + w < dist[v]:

                dist[v] = dist[u] + w

                changed = True

        if not changed:
            break

    for u, v, w in edges:

        if (
            dist[u] != INF
            and dist[u] + w < dist[v]
        ):

            return None

    return dist


# ------------------------------------------------------------
# 4.5 Floyd-Warshall
# All-Pairs Shortest Path
# Time: O(V^3)
# Space: O(V^2)
# ------------------------------------------------------------

def floyd_warshall(n, edges, directed=True):

    INF = float("inf")

    dist = [
        [INF] * n
        for _ in range(n)
    ]

    for i in range(n):

        dist[i][i] = 0

    for u, v, w in edges:

        dist[u][v] = min(
            dist[u][v],
            w
        )

        if not directed:

            dist[v][u] = min(
                dist[v][u],
                w
            )

    for k in range(n):

        for i in range(n):

            if dist[i][k] == INF:
                continue

            for j in range(n):

                if dist[k][j] == INF:
                    continue

                nd = (
                    dist[i][k]
                    +
                    dist[k][j]
                )

                if nd < dist[i][j]:

                    dist[i][j] = nd

    return dist


# ------------------------------------------------------------
# 4.6 A* - Grid
#
# grid:
# "." passable
# "*" obstacle
#
# start = (x, y)
# target = (x, y)
#
# Time depends on heuristic
# ------------------------------------------------------------

def astar_grid(grid, start, target):

    n = len(grid)
    m = len(grid[0])

    sx, sy = start
    tx, ty = target

    D = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    def heuristic(x, y):

        return (
            abs(x - tx)
            +
            abs(y - ty)
        )

    INF = float("inf")

    dist = {
        start: 0
    }

    pq = [
        (
            heuristic(sx, sy),
            0,
            sx,
            sy
        )
    ]

    while pq:

        f, g, x, y = heapq.heappop(pq)

        if (x, y) == target:

            return g

        if g != dist.get(
            (x, y),
            INF
        ):
            continue

        for dx, dy in D:

            nx = x + dx
            ny = y + dy

            if not (
                0 <= nx < n
                and
                0 <= ny < m
            ):
                continue

            if grid[nx][ny] == "*":
                continue

            ng = g + 1

            if ng < dist.get(
                (nx, ny),
                INF
            ):

                dist[(nx, ny)] = ng

                nf = (
                    ng
                    +
                    heuristic(nx, ny)
                )

                heapq.heappush(
                    pq,
                    (
                        nf,
                        ng,
                        nx,
                        ny
                    )
                )

    return -1


# ============================================================
# 5. MINIMUM SPANNING TREE
# ============================================================

# ------------------------------------------------------------
# 5.1 Kruskal
# Time: O(E log E)
#
# edges = [(u, v, w), ...]
# ------------------------------------------------------------

def kruskal(n, edges):

    edges = sorted(
        edges,
        key=lambda x: x[2]
    )

    dsu = DSU(n)

    total = 0

    used = 0

    mst = []

    for u, v, w in edges:

        if dsu.union(u, v):

            total += w

            used += 1

            mst.append(
                (u, v, w)
            )

            if used == n - 1:
                break

    if used != n - 1:

        return None

    return total, mst


# ------------------------------------------------------------
# 5.2 Prim
# Time: O(E log V)
# graph[u] = [(v, weight), ...]
# ------------------------------------------------------------

def prim(graph, start=0):

    n = len(graph)

    visited = [False] * n

    pq = [
        (0, start, -1)
    ]

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

            mst.append(
                (
                    parent,
                    u,
                    w
                )
            )

        for v, cost in graph[u]:

            if not visited[v]:

                heapq.heappush(
                    pq,
                    (
                        cost,
                        v,
                        u
                    )
                )

    if used != n:

        return None

    return total, mst


# ============================================================
# 6. DAG
# ============================================================

# ------------------------------------------------------------
# 6.1 Topological Sort - Kahn BFS
#
# return [] => cycle exists
# ------------------------------------------------------------

def topological_sort(graph):

    n = len(graph)

    indegree = [0] * n

    for u in range(n):

        for v in graph[u]:

            indegree[v] += 1

    q = deque()

    for i in range(n):

        if indegree[i] == 0:

            q.append(i)

    order = []

    while q:

        u = q.popleft()

        order.append(u)

        for v in graph[u]:

            indegree[v] -= 1

            if indegree[v] == 0:

                q.append(v)

    if len(order) != n:

        return []

    return order


# ------------------------------------------------------------
# 6.2 DAG DP - Longest Path
#
# graph[u] = [(v, weight), ...]
# DAG only
# ------------------------------------------------------------

def dag_longest_path(graph, start):

    n = len(graph)

    simple_graph = [
        [] for _ in range(n)
    ]

    for u in range(n):

        for v, w in graph[u]:

            simple_graph[u].append(v)

    order = topological_sort(
        simple_graph
    )

    if not order:

        return None

    NEG_INF = float("-inf")

    dist = [NEG_INF] * n

    dist[start] = 0

    for u in order:

        if dist[u] == NEG_INF:
            continue

        for v, w in graph[u]:

            dist[v] = max(
                dist[v],
                dist[u] + w
            )

    return dist


# ============================================================
# 7. CYCLE DETECTION
# ============================================================

# ------------------------------------------------------------
# 7.1 Undirected Cycle - DFS
# ------------------------------------------------------------

def has_cycle_undirected(graph):

    n = len(graph)

    visited = [False] * n

    def dfs(u, parent):

        visited[u] = True

        for v in graph[u]:

            if not visited[v]:

                if dfs(v, u):

                    return True

            elif v != parent:

                return True

        return False

    for i in range(n):

        if not visited[i]:

            if dfs(i, -1):

                return True

    return False


# ------------------------------------------------------------
# 7.2 Undirected Cycle - Union Find
# ------------------------------------------------------------

def has_cycle_union_find(n, edges):

    dsu = DSU(n)

    for u, v in edges:

        if not dsu.union(u, v):

            return True

    return False


# ------------------------------------------------------------
# 7.3 Directed Cycle - DFS Coloring
#
# state:
# 0 = unvisited
# 1 = visiting
# 2 = finished
# ------------------------------------------------------------

def has_cycle_directed(graph):

    n = len(graph)

    state = [0] * n

    def dfs(u):

        state[u] = 1

        for v in graph[u]:

            if state[v] == 1:

                return True

            if state[v] == 0:

                if dfs(v):

                    return True

        state[u] = 2

        return False

    for u in range(n):

        if state[u] == 0:

            if dfs(u):

                return True

    return False


# ============================================================
# 8. NETWORK FLOW
# ============================================================

# ------------------------------------------------------------
# 8.1 Dinic Max Flow
# ------------------------------------------------------------

class Dinic:

    class Edge:

        def __init__(self, to, rev, cap):

            self.to = to
            self.rev = rev
            self.cap = cap


    def __init__(self, n):

        self.n = n

        self.graph = [
            [] for _ in range(n)
        ]


    def add_edge(self, u, v, cap):

        forward = self.Edge(
            v,
            len(self.graph[v]),
            cap
        )

        backward = self.Edge(
            u,
            len(self.graph[u]),
            0
        )

        self.graph[u].append(
            forward
        )

        self.graph[v].append(
            backward
        )


    def bfs(self, s, t):

        self.level = [-1] * self.n

        q = deque([s])

        self.level[s] = 0

        while q:

            u = q.popleft()

            for e in self.graph[u]:

                if (
                    e.cap > 0
                    and
                    self.level[e.to] == -1
                ):

                    self.level[e.to] = (
                        self.level[u]
                        +
                        1
                    )

                    q.append(e.to)

        return self.level[t] != -1


    def dfs(self, u, t, pushed):

        if pushed == 0:
            return 0

        if u == t:
            return pushed

        while self.it[u] < len(
            self.graph[u]
        ):

            e = self.graph[u][
                self.it[u]
            ]

            if (
                e.cap > 0
                and
                self.level[e.to]
                ==
                self.level[u] + 1
            ):

                flow = self.dfs(
                    e.to,
                    t,
                    min(
                        pushed,
                        e.cap
                    )
                )

                if flow:

                    e.cap -= flow

                    rev_edge = (
                        self.graph[e.to][
                            e.rev
                        ]
                    )

                    rev_edge.cap += flow

                    return flow

            self.it[u] += 1

        return 0


    def max_flow(self, s, t):

        flow = 0

        INF = 10**18

        while self.bfs(s, t):

            self.it = [0] * self.n

            while True:

                pushed = self.dfs(
                    s,
                    t,
                    INF
                )

                if pushed == 0:
                    break

                flow += pushed

        return flow


# ============================================================
# 9. MATCHING
# ============================================================

# ------------------------------------------------------------
# 9.1 Bipartite Matching - DFS Augmenting Path
#
# graph[u] contains right-side vertices
#
# left side:
# 0 ... left_n - 1
#
# right side:
# 0 ... right_n - 1
# ------------------------------------------------------------

def bipartite_matching(
    left_n,
    right_n,
    graph
):

    match_right = [-1] * right_n

    def dfs(u, visited):

        for v in graph[u]:

            if visited[v]:
                continue

            visited[v] = True

            if (
                match_right[v] == -1
                or
                dfs(
                    match_right[v],
                    visited
                )
            ):

                match_right[v] = u

                return True

        return False

    matching = 0

    for u in range(left_n):

        visited = [False] * right_n

        if dfs(u, visited):

            matching += 1

    return matching, match_right


# ============================================================
# 10. GRID GRAPH TEMPLATES
# ============================================================

# ------------------------------------------------------------
# 10.1 Grid BFS Shortest Path
# ------------------------------------------------------------

def grid_bfs(
    grid,
    start
):

    n = len(grid)
    m = len(grid[0])

    sx, sy = start

    dist = [
        [-1] * m
        for _ in range(n)
    ]

    dist[sx][sy] = 0

    q = deque([
        (sx, sy)
    ])

    D = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while q:

        x, y = q.popleft()

        for dx, dy in D:

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < n
                and
                0 <= ny < m
                and
                grid[nx][ny] != "*"
                and
                dist[nx][ny] == -1
            ):

                dist[nx][ny] = (
                    dist[x][y]
                    +
                    1
                )

                q.append(
                    (nx, ny)
                )

    return dist


# ------------------------------------------------------------
# 10.2 Grid Connected Components
# ------------------------------------------------------------

def grid_components(grid):

    n = len(grid)
    m = len(grid[0])

    visited = [
        [False] * m
        for _ in range(n)
    ]

    D = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    components = []

    for i in range(n):

        for j in range(m):

            if grid[i][j] == "*":
                continue

            if visited[i][j]:
                continue

            q = deque([
                (i, j)
            ])

            visited[i][j] = True

            component = []

            while q:

                x, y = q.popleft()

                component.append(
                    (x, y)
                )

                for dx, dy in D:

                    nx = x + dx
                    ny = y + dy

                    if (
                        0 <= nx < n
                        and
                        0 <= ny < m
                        and
                        grid[nx][ny] != "*"
                        and
                        not visited[nx][ny]
                    ):

                        visited[nx][ny] = True

                        q.append(
                            (nx, ny)
                        )

            components.append(
                component
            )

    return components


# ============================================================
# CHEAT SHEET
# ============================================================

"""
============================================================
SHORTEST PATH SELECTION
============================================================

1. Every edge weight = 1
        -> BFS
        -> O(V + E)

2. Edge weight only 0 / 1
        -> 0-1 BFS
        -> O(V + E)

3. Edge weight >= 0
        -> Dijkstra
        -> O((V + E) log V)

4. Negative edges
        -> Bellman-Ford
        -> O(VE)

5. All-pairs shortest path
        -> Floyd-Warshall
        -> O(V^3)

6. One target + good heuristic
        -> A*
        -> priority = g + h


============================================================
GRAPH ALGORITHM SELECTION
============================================================

Traversal
    -> BFS / DFS

Connected Components
    -> BFS / DFS / DSU

Unweighted Shortest Path
    -> BFS

Weighted Shortest Path
    -> Dijkstra

Negative Weight
    -> Bellman-Ford

All Pairs
    -> Floyd-Warshall

Minimum Spanning Tree
    -> Kruskal / Prim

Cycle Detection
    -> DFS / Topological Sort / DSU

DAG
    -> Topological Sort + DP

Strongly Connected Components
    -> Tarjan / Kosaraju

Max Flow
    -> Dinic

Bipartite Matching
    -> Augmenting Path / Hopcroft-Karp


============================================================
YOUR SUPERMARKET PROBLEM
============================================================

Grid Graph
    +
Connected Components
    +
BFS Shortest Path
    +
Distance Accumulation

NOT:
    Dijkstra
    A*
    MST
"""