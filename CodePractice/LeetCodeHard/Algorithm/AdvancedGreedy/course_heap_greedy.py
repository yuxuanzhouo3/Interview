"""
3. 课程表 III — 期限 + 时长反悔堆
App: LC 630。
"""

import heapq


def schedule_course(courses):
    """courses[i] = [duration, lastDay]"""
    courses = sorted(courses, key=lambda x: x[1])
    pq = []  # max-heap of durations via negative
    time = 0
    for d, last in courses:
        if time + d <= last:
            heapq.heappush(pq, -d)
            time += d
        elif pq and -pq[0] > d:
            time += d + heapq.heappop(pq)
            heapq.heappush(pq, -d)
    return len(pq)


def _test():
    courses = [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
    assert schedule_course(courses) == 3
    assert schedule_course([[1, 2], [2, 3]]) == 2
    print("course_heap_greedy OK", schedule_course(courses))


APPLICATION = """
- 按截止期排序；超期则丢掉目前最长的课（反悔）。
- 堆维护已选课程时长。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
