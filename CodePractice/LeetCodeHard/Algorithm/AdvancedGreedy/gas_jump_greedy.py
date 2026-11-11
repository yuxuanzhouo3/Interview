"""
6. 环形加油站 / 跳跃可达
App: LC 134 / 55 / 45。
"""


def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost):
        return -1
    start = tank = 0
    for i, (g, c) in enumerate(zip(gas, cost)):
        tank += g - c
        if tank < 0:
            start = i + 1
            tank = 0
    return start


def can_jump(nums):
    far = 0
    for i, x in enumerate(nums):
        if i > far:
            return False
        far = max(far, i + x)
    return True


def jump_min(nums):
    ans = cur = far = 0
    for i in range(len(nums) - 1):
        far = max(far, i + nums[i])
        if i == cur:
            ans += 1
            cur = far
    return ans


def _test():
    assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert can_jump([2, 3, 1, 1, 4]) is True
    assert can_jump([3, 2, 1, 0, 4]) is False
    assert jump_min([2, 3, 1, 1, 4]) == 2
    print("gas_jump_greedy OK")


APPLICATION = """
- 134: 总油够则必有解；从亏空重置点的下一点出发。
- 55: 维护最远可达；45: 层内扩展最远。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
