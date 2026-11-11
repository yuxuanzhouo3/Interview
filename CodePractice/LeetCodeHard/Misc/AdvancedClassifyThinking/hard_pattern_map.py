"""
ClassifyThinking / Hard 模式地图
Template: query(pattern→path) / update / add / del
App: 复习索引，链到 Algorithm / DataStructure 文件。
"""


class HardPatternMap:
    def __init__(self):
        self.map = {
            "min_max_binary": "Misc/.../answer_binary_search.py",
            "sliding_window": "Algorithm/AdvancedWindow/",
            "interval_greedy": "Algorithm/AdvancedOO/",
            "union_find": "Algorithm/AdvancedUF/ + DataStructure/.../union_find.py",
            "shortest_path": "DataStructure/AdvancedGraph/shortest_path.py",
            "max_flow": "DataStructure/AdvancedGraph/network_flow.py",
            "bipartite_match": "DataStructure/AdvancedGraph/bipartite.py",
            "interval_dp": "Algorithm/AdvancedDP/interval_dp.py",
            "bitmask_dp": "Algorithm/AdvancedDP/bitmask_dp.py",
            "string_match": "DataStructure/AdvancedString/",
            "lru_lfu": "Misc/AdvancedCodeSkill/lru_design.py",
            "meet_in_middle": "Misc/.../meet_in_middle.py",
            "offline_dsu": "Misc/.../offline_query.py",
        }

    def add(self, pattern, path):
        self.map[pattern] = path

    def update(self, pattern, path):
        if pattern not in self.map:
            return False
        self.map[pattern] = path
        return True

    def delete(self, pattern):
        self.map.pop(pattern, None)

    def query(self, pattern=None):
        if pattern is None:
            return dict(self.map)
        return self.map.get(pattern, "unknown — use type_router")


def _test():
    m = HardPatternMap()
    assert "network_flow" in m.query("max_flow")
    m.add("bloom", "Misc/AdvancedCodeSkill/bloom_filter.py")
    assert m.update("bloom", "Misc/AdvancedCodeSkill/bloom_filter.py")
    m.delete("bloom")
    assert m.query("bloom") == "unknown — use type_router"
    print("PatternMap OK", len(m.query()))


APPLICATION = """
- 周复习：按 pattern 扫地图，每项对应写一道 Hard。
- 与 type_router 配合：router 找类型，map 找文件。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
