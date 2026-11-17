"""
LC 2675. Array of Objects to Matrix — Hard (JS)
Best: Flatten keys; collect columns — O(n * keys)
"""

from typing import List, Any


class Solution:
    def jsonToMatrix(self, arr: List[Any]) -> List[List[Any]]:
        def flat(obj, prefix=""):
            out = {}
            if isinstance(obj, dict):
                if not obj:
                    out[prefix] = obj if prefix else {}
                for k, v in obj.items():
                    np = f"{prefix}.{k}" if prefix else k
                    out.update(flat(v, np))
            elif isinstance(obj, list):
                if not obj:
                    out[prefix] = []
                for i, v in enumerate(obj):
                    np = f"{prefix}.{i}" if prefix else str(i)
                    out.update(flat(v, np))
            else:
                out[prefix] = obj
            return out

        rows = [flat(o) for o in arr]
        cols = sorted({k for r in rows for k in r})
        mat = [cols]
        for r in rows:
            mat.append([r.get(c, "") for c in cols])
        return mat


def _test():
    s = Solution()
    m = s.jsonToMatrix([{"a": 1, "b": 2}, {"c": 3}])
    assert m[0] == ["a", "b", "c"]
    print("LC 2675 OK")


if __name__ == "__main__":
    _test()
