"""
2. 最少区间覆盖 / 跳跃浇水
Template: query(min_count) 
App: LC 1326 / 45 Jump Game II。
"""


def video_stitching(clips, time):
    """LC 1024 思想：覆盖 [0,time]。"""
    clips = sorted(clips)
    ans = cur_end = next_end = i = 0
    n = len(clips)
    while cur_end < time:
        while i < n and clips[i][0] <= cur_end:
            next_end = max(next_end, clips[i][1])
            i += 1
        if next_end == cur_end:
            return -1
        ans += 1
        cur_end = next_end
    return ans


def jump_ii(nums):
    """每次在当前可达范围内扩展最远。"""
    n = len(nums)
    ans = cur_end = far = 0
    for i in range(n - 1):
        far = max(far, i + nums[i])
        if i == cur_end:
            ans += 1
            cur_end = far
    return ans


def _test():
    assert video_stitching([[0, 2], [4, 6], [0, 4], [1, 5]], 5) in (2, 3)
    assert jump_ii([2, 3, 1, 1, 4]) == 2
    assert jump_ii([2, 3, 0, 1, 4]) == 2
    print("interval_cover OK", jump_ii([2, 3, 1, 1, 4]))


APPLICATION = """
- 断点处必须选「能延伸最远」的一段。
- 1326 水龙头：先化成覆盖区间再同套路。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
