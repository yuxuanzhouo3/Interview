"""
LC 2296. Design a Text Editor — Hard
https://leetcode.com/problems/design-a-text-editor/
Best: two stacks left/right of cursor — O(1) amortized per char
"""


class TextEditor:
    def __init__(self):
        self.left: list[str] = []
        self.right: list[str] = []

    def addText(self, text: str) -> None:
        self.left.extend(text)

    def deleteText(self, k: int) -> int:
        deleted = min(k, len(self.left))
        for _ in range(deleted):
            self.left.pop()
        return deleted

    def cursorLeft(self, k: int) -> str:
        while self.left and k > 0:
            self.right.append(self.left.pop())
            k -= 1
        return self._prefix()

    def cursorRight(self, k: int) -> str:
        while self.right and k > 0:
            self.left.append(self.right.pop())
            k -= 1
        return self._prefix()

    def _prefix(self) -> str:
        return "".join(self.left[-10:])


def _test():
    te = TextEditor()
    te.addText("leetcode")
    assert te.deleteText(4) == 4
    te.addText("practice")
    assert te.cursorRight(3) == "etpractice"
    assert te.cursorLeft(8) == "leet"
    assert te.deleteText(10) == 4
    assert te.cursorLeft(2) == ""
    assert te.cursorRight(6) == "practi"
    print("LC 2296 OK")


if __name__ == "__main__":
    _test()
