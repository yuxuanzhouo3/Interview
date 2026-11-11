"""
AdvancedGraph / Floyd-Warshall（全源最短路）
Template: query(i,j) / update(edge) / add / del
App: 稠密小图全源；传递闭包；LC 1334 / 1462 类。
来源: CodePractice/graph.py §4.5
"""


class FloydWarshall:
    def __init__(self, n=0):
        self.n = n
        self.INF = float("inf")
        self.w = [[self.INF] * n for _ in range(n)]
        for i in range(n):
            self.w[i][i] = 0

    def add(self, u, v=None, w=0):
        """扩点 add(u)；或 add(u,v,w)。"""
        if v is None:
            self._ensure(u + 1)
            return
        self._ensure(max(u, v) + 1)
        self.w[u][v] = min(self.w[u][v], w)

    def _ensure(self, n):
        while self.n < n:
            for row in self.w:
                row.append(self.INF)
            self.w.append([self.INF] * (self.n + 1))
            self.w[self.n][self.n] = 0
            self.n += 1

    def update(self, u, v, w):
        if u >= self.n or v >= self.n:
            return False
        self.w[u][v] = w
        return True

    def delete(self, u, v):
        if u < self.n and v < self.n and u != v:
            self.w[u][v] = self.INF

    def query(self, i=None, j=None):
        dist = self.run()
        if i is None:
            return dist
        if j is None:
            return dist[i]
        return dist[i][j]

    def run(self):
        dist = [row[:] for row in self.w]
        for k in range(self.n):
            for i in range(self.n):
                if dist[i][k] == self.INF:
                    continue
                for j in range(self.n):
                    if dist[k][j] == self.INF:
                        continue
                    nd = dist[i][k] + dist[k][j]
                    if nd < dist[i][j]:
                        dist[i][j] = nd
        return dist


def _test():
    g = FloydWarshall(4)
    for e in [(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 6), (2, 3, 3)]:
        g.add(*e)
    assert g.query(0, 3) == 6
    assert g.update(1, 3, 1)
    assert g.query(0, 3) == 2
    g.delete(1, 3)
    assert g.query(0, 3) == 6
    print("Floyd OK", g.query(0, 3))


APPLICATION = """
- 全源 O(V³)；V≲400。传递闭包把边权当可达即可。
- 负环: dist[i][i] < 0。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
