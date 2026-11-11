"""
1. 最多不重叠区间（按结束时间贪心）
Template: query(count|kept) / — / add(intervals) / —
App: LC 435/452 思想；会议室选择。
"""


def erase_overlap_count(intervals):
    """最少删除数 = n - 最多保留不重叠。"""
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda x: x[1])
    keep, end = 0, float("-inf")
    for s, e in intervals:
        if s >= end:
            keep += 1
            end = e
    return len(intervals) - keep


def max_non_overlap(intervals):
    intervals = sorted(intervals, key=lambda x: x[1])
    keep, end = [], float("-inf")
    for seg in intervals:
        s, e = seg
        if s >= end:
            keep.append(seg)
            end = e
    return keep


def _test():
    iv = [[1, 2], [2, 3], [3, 4], [1, 3]]
    assert erase_overlap_count(iv) == 1
    assert max_non_overlap([[1, 4], [2, 3], [3, 5]]) == [[2, 3], [3, 5]] or \
           len(max_non_overlap([[1, 4], [2, 3], [3, 5]])) == 2
    print("interval_select OK", max_non_overlap(iv))


APPLICATION = """
- 按结束时间升序，能选就选 → 最多不重叠。
- 证明：交换论证，结束越早留给后面空间越大。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
