"""
CodeSkill / 回溯 + 剪枝
Template: query(solutions) / update(n) / add(constraint) / del
App: N 皇后简化；LC 51 思想。
"""


class NQueens:
    def __init__(self, n=0):
        self.n = n
        self.forbid_cols = set()  # 额外禁列（演示 add/del）

    def add(self, col_forbid=None):
        if col_forbid is not None:
            self.forbid_cols.add(col_forbid)

    def update(self, n):
        self.n = n
        return True

    def delete(self, col_forbid=None):
        if col_forbid is None:
            self.forbid_cols.clear()
        else:
            self.forbid_cols.discard(col_forbid)

    def query(self):
        n = self.n
        cols, d1, d2 = set(), set(), set()
        board = [["."] * n for _ in range(n)]
        res = []

        def dfs(r):
            if r == n:
                res.append(["".join(row) for row in board])
                return
            for c in range(n):
                if c in self.forbid_cols or c in cols or (r - c) in d1 or (r + c) in d2:
                    continue
                cols.add(c)
                d1.add(r - c)
                d2.add(r + c)
                board[r][c] = "Q"
                dfs(r + 1)
                board[r][c] = "."
                cols.remove(c)
                d1.remove(r - c)
                d2.remove(r + c)

        dfs(0)
        return res


def _test():
    q = NQueens()
    q.update(4)
    sols = q.query()
    assert len(sols) == 2
    q.add(0)  # 禁第 0 列 → 可能变少
    assert len(q.query()) <= 2
    q.delete(0)
    assert len(q.query()) == 2
    print("Backtrack OK", len(sols))


APPLICATION = """
- 改状态 → 递归 → 撤销；剪枝条件写在扩展前。
- 数独/N皇后：行、列、块（对角）占用集合。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
