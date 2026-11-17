"""
1. 入门 DP · 爬楼梯 / 打家劫舍 / 最大子数组和
对照: 灵茶山 DP 题单 §一
App: LC 70 / 746 / 198 / 213 / 53 / 918
合并自: counting_dp.climb_stairs / linear_state_dp.house_robber / subsequence_dp.max_subarray
"""


def climb_stairs(n):
    """爬楼梯。LC 70。"""
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def min_cost_climbing_stairs(cost):
    """最小花费爬楼梯。LC 746。"""
    a = b = 0
    for c in cost:
        a, b = b, c + min(a, b)
    return min(a, b)


def house_robber(nums):
    """打家劫舍：相邻不偷。LC 198。"""
    take = skip = 0
    for x in nums:
        take, skip = skip + x, max(skip, take)
    return max(take, skip)


def house_robber_ii(nums):
    """环形打家劫舍。LC 213。"""
    if len(nums) == 1:
        return nums[0]

    def rob_range(lo, hi):
        take = skip = 0
        for i in range(lo, hi):
            take, skip = skip + nums[i], max(skip, take)
        return max(take, skip)

    return max(rob_range(0, len(nums) - 1), rob_range(1, len(nums)))


def max_subarray(nums):
    """Kadane 最大子数组和。LC 53。f[i]=max(f[i-1],0)+a[i]，答案 max(f)。"""
    cur = res = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        res = max(res, cur)
    return res


def max_subarray_circular(nums):
    """环形最大子数组和。LC 918。max(普通最大, total-最小子数组)。"""
    total = sum(nums)
    cur_max = best_max = nums[0]
    cur_min = best_min = nums[0]
    for x in nums[1:]:
        cur_max = max(x, cur_max + x)
        best_max = max(best_max, cur_max)
        cur_min = min(x, cur_min + x)
        best_min = min(best_min, cur_min)
    if best_max < 0:
        return best_max
    return max(best_max, total - best_min)


def _test():
    assert climb_stairs(5) == 8
    assert min_cost_climbing_stairs([10, 15, 20]) == 15
    assert house_robber([2, 7, 9, 3, 1]) == 12
    assert house_robber_ii([2, 3, 2]) == 3
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray_circular([5, -3, 5]) == 10
    print("1.intro_dp OK")


APPLICATION = """
- 思考「最后一步」：爬楼梯看最后一/两级；劫舍看偷或不偷。
- 返回 f[n]（恰好到终点）还是 max(f)（任意结尾）？子数组类常是后者。
- 环形：拆成「不用首」或「不用尾」，或 total-最小段。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
