"""
LC 588. Design In-Memory File System — Hard
https://leetcode.com/problems/design-in-memory-file-system/
Best: Trie of directories / files — O(path)
"""

from typing import List


class FileSystem:
    def __init__(self):
        self.root = {"dirs": {}, "files": {}}  # files: name -> content

    def _traverse(self, path: str, create_dirs: bool = False):
        if path == "/":
            return self.root
        node = self.root
        parts = [p for p in path.split("/") if p]
        for i, p in enumerate(parts):
            is_last = i == len(parts) - 1
            if p in node["files"] and not is_last:
                return None
            if p in node["files"]:
                return node  # parent of file; caller handles
            if p not in node["dirs"]:
                if create_dirs:
                    node["dirs"][p] = {"dirs": {}, "files": {}}
                else:
                    return None
            node = node["dirs"][p]
        return node

    def ls(self, path: str) -> List[str]:
        if path == "/":
            names = list(self.root["dirs"].keys()) + list(self.root["files"].keys())
            return sorted(names)
        parts = [p for p in path.split("/") if p]
        node = self.root
        for i, p in enumerate(parts):
            if p in node["files"]:
                return [p]  # file path → only file name
            node = node["dirs"][p]
        return sorted(list(node["dirs"].keys()) + list(node["files"].keys()))

    def mkdir(self, path: str) -> None:
        self._traverse(path, create_dirs=True)

    def addContentToFile(self, filePath: str, content: str) -> None:
        parts = [p for p in filePath.split("/") if p]
        parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        parent = self._traverse(parent_path, create_dirs=True)
        name = parts[-1]
        parent["files"][name] = parent["files"].get(name, "") + content

    def readContentFromFile(self, filePath: str) -> str:
        parts = [p for p in filePath.split("/") if p]
        parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        parent = self._traverse(parent_path)
        return parent["files"][parts[-1]]


class Solution:
    def run(self, ops, args):
        fs = FileSystem()
        out = []
        for op, a in zip(ops, args):
            if op == "ls":
                out.append(fs.ls(a[0]))
            elif op == "mkdir":
                fs.mkdir(a[0]); out.append(None)
            elif op == "addContentToFile":
                fs.addContentToFile(a[0], a[1]); out.append(None)
            elif op == "readContentFromFile":
                out.append(fs.readContentFromFile(a[0]))
        return out


def _test():
    fs = FileSystem()
    assert fs.ls("/") == []
    fs.mkdir("/a/b/c")
    fs.addContentToFile("/a/b/c/d", "hello")
    assert fs.ls("/") == ["a"]
    assert fs.readContentFromFile("/a/b/c/d") == "hello"
    fs.addContentToFile("/a/b/c/d", " world")
    assert fs.readContentFromFile("/a/b/c/d") == "hello world"
    assert fs.ls("/a/b/c/d") == ["d"]
    print("LC 588 OK")


if __name__ == "__main__":
    _test()
