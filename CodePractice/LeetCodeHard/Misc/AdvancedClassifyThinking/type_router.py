"""
ClassifyThinking / 题型路由
Template: query(route) / update(rule) / add(keyword→type) / del
App: Hard 开题 30 秒决策；对照 Algorithm / DataStructure 目录。
"""


class TypeRouter:
    def __init__(self):
        # keyword -> type
        self.rules = {}
        self._default = [
            ("shortest", "graph_shortest"),
            ("path", "graph_or_dp"),
            ("island", "grid_uf_or_bfs"),
            ("subarray", "window_or_prefix"),
            ("substring", "window_or_string"),
            ("interval", "greedy_or_interval_dp"),
            ("schedule", "greedy_heap"),
            ("match", "dp_or_kmp"),
            ("palindrome", "dp_or_manacher"),
            ("tree", "tree_dp_or_lca"),
            ("cache", "design_lru"),
            ("stream", "heap_or_fenwick"),
            ("mod", "math_or_counting_dp"),
            ("minimize the maximum", "answer_binary_search"),
            ("at most k", "window_at_most"),
        ]
        for k, t in self._default:
            self.rules[k] = t

    def add(self, keyword, typ):
        self.rules[keyword.lower()] = typ

    def update(self, keyword, typ):
        if keyword.lower() not in self.rules:
            return False
        self.rules[keyword.lower()] = typ
        return True

    def delete(self, keyword):
        self.rules.pop(keyword.lower(), None)

    def query(self, text):
        """返回命中的 (keyword, type) 列表，按关键词长度降序。"""
        s = text.lower()
        hits = [(k, t) for k, t in self.rules.items() if k in s]
        hits.sort(key=lambda x: -len(x[0]))
        return hits or [("unknown", "read_statement_again")]


def _test():
    r = TypeRouter()
    h = r.query("Minimize the maximum capacity of ship")
    assert h[0][1] == "answer_binary_search"
    r.add("critical connections", "bridge_tarjan")
    assert r.query("critical connections in network")[0][1] == "bridge_tarjan"
    assert r.update("island", "grid_dfs")
    r.delete("island")
    assert "island" not in dict(r.query("island count"))
    print("TypeRouter OK", r.query("subarray sum at most k")[:2])


APPLICATION = """
- 先关键词路由，再打开对应 Advanced* 模板，避免一上来写错范式。
- 「最小化最大值 / 最大化最小值」几乎总是答案二分。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
