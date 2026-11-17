"""
LC 2753. Count Houses in a Circular Street II — Hard
Best: Street API open doors walk — O(n)
"""

# Street API mock
class Street:
    def __init__(self, doors):
        self.doors = doors
        self.i = 0
        self.n = len(doors)
    def openDoor(self):
        self.doors[self.i] = 1
    def closeDoor(self):
        self.doors[self.i] = 0
    def isDoorOpen(self):
        return self.doors[self.i] == 1
    def moveRight(self):
        self.i = (self.i + 1) % self.n
    def moveLeft(self):
        self.i = (self.i - 1) % self.n


class Solution:
    def houseCount(self, street: "Street", k: int) -> int:
        # at least one open; close all then open one as marker
        for _ in range(k):
            street.closeDoor()
            street.moveRight()
        street.openDoor()
        street.moveRight()
        ans = 1
        while not street.isDoorOpen():
            ans += 1
            street.moveRight()
        return ans


def _test():
    st = Street([1, 0, 1, 0, 0])
    # move to an open door first
    while not st.isDoorOpen():
        st.moveRight()
    assert Solution().houseCount(st, 5) == 5
    print("LC 2753 OK")


if __name__ == "__main__":
    _test()
