"""
LC 2691. Immortality Helper — Hard (JS)
Best: Proxy deep immutable wrapper (Python demo)
"""

class Immutable:
    def __init__(self, obj):
        self._obj = obj

    def get(self, path=None):
        cur = self._obj
        if not path:
            return cur
        for p in path.split("."):
            if isinstance(cur, list):
                cur = cur[int(p)]
            else:
                cur = cur[p]
        return cur

    def set(self, path, value):
        import copy
        new = copy.deepcopy(self._obj)
        parts = path.split(".")
        cur = new
        for p in parts[:-1]:
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        last = parts[-1]
        if isinstance(cur, list):
            cur[int(last)] = value
        else:
            cur[last] = value
        return Immutable(new)


def _test():
    im = Immutable({"x": 1, "y": [2, 3]})
    im2 = im.set("y.0", 9)
    assert im.get("y.0") == 2 and im2.get("y.0") == 9
    print("LC 2691 OK")


if __name__ == "__main__":
    _test()
