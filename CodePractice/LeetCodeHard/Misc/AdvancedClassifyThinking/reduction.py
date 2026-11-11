"""
ClassifyThinking / 问题转化（建模）
Template: query(model) / update / add(constraint) / del
App: Hard 常「不像算法题」→ 建图/流/差分约束后再套模板。
"""


class ReductionNotebook:
    def __init__(self):
        self.constraints = []  # str
        self.model = "none"

    def add(self, constraint):
        self.constraints.append(constraint)
        self._infer()

    def update(self, idx, constraint):
        if not (0 <= idx < len(self.constraints)):
            return False
        self.constraints[idx] = constraint
        self._infer()
        return True

    def delete(self, idx=None):
        if idx is None:
            self.constraints.clear()
        elif 0 <= idx < len(self.constraints):
            self.constraints.pop(idx)
        self._infer()

    def _infer(self):
        text = " ".join(self.constraints).lower()
        if any(k in text for k in ("capacity", "cut", "flow", "matching")):
            self.model = "network_flow"
        elif any(k in text for k in ("same component", "merge", "connect")):
            self.model = "union_find"
        elif "difference" in text or "at least" in text and "than" in text:
            self.model = "diff_constraint_bellman"
        elif "dependency" in text or "order" in text:
            self.model = "topo_dag"
        elif self.constraints:
            self.model = "rephrase_then_retry"
        else:
            self.model = "none"

    def query(self):
        return self.model, list(self.constraints)


def _test():
    r = ReductionNotebook()
    r.add("tasks have capacity limits on workers")
    r.add("need max assignments matching")
    assert r.query()[0] == "network_flow"
    assert r.update(0, "accounts should merge if same email")
    # still has matching → flow；再删 matching
    r.delete(1)
    assert r.query()[0] == "union_find"
    r.delete()
    assert r.query()[0] == "none"
    print("Reduction OK", r.query())


APPLICATION = """
- 转化表: 匹配→流/二分图；连通→DSU；不等式组→差分约束最短路。
- 面试先说「化成什么已知问题」，再写代码。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
