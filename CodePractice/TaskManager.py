'''
🧠 第一轮：设计一个 TaskManager 类（带优先级和 task_id 排序）
✅ 要求功能

add(user_id, task_id, priority)   # 添加任务
update(task_id, priority)         # 更新任务优先级
remove(task_id)                   # 删除任务
exeTop()                          # 弹出优先级最高任务
✅ 排序要求
优先级大的排前面
优先级相同时，task_id 较大的排前（注意：字符串 "task12" > "task3"）
'''
import heapq
from collections import defaultdict
from heapq import heappop
from typing import List


# import heapq
#
# class TaskManager(object):
#     def __init__(self):
#         self.heap = []
#         self.d = {}
#
#     def add(self, user_id, task_id, priority):  # 添加任务
#         heapq.heappush(self.heap, (-priority, -int(task_id[4:]), task_id))
#         self.d[task_id] = [priority, 0]
#
#     def update(self,task_id, priority):  # 更新任务优先级
#         self.d[task_id] = [priority, 0]
#
#     def remove(self, task_id):  # 删除任务
#         if task_id in self.d:
#             self.d[task_id][1] = 1
#
#     def exeTop(self):  # 弹出优先级最高任务
#         if self.heap:
#             _, _, task_id = heapq.heappop(self.heap)
#             while self.heap and self.d[task_id][1] == 1:
#                 _, _, task_id = heapq.heappop(self.heap)
#             return task_id
#         return None

# tm = TaskManager()
# tm.add("u1", "task1", 10)
# tm.add("u2", "task2", 20)
# tm.add("u3", "task12", 20)
#
# print(tm.exeTop())  # task12 (priority 相同但 task12 > task2)
# print(tm.exeTop())  # task2
# tm.update("task1", 25)
# print(tm.exeTop())  # task1
# print(tm.exeTop())  # None


class TaskManager:

    def __init__(self, tasks: List[List[int]]):
        self.task_user_priority = defaultdict(list)
        self.task_priority = []

        for user, task, priority in tasks:
            self.task_user_priority[task] = [user, priority]
            heapq.heappush(self.task_priority, [-priority, -task])

    def add(self, userId: int, taskId: int, priority: int) -> None:

        self.task_user_priority[taskId] = [userId, priority]
        heapq.heappush(self.task_priority, [-priority, -taskId])

    def edit(self, taskId: int, newPriority: int) -> None:

        self.task_user_priority[taskId][1] = newPriority
        heapq.heappush(self.task_priority, [-newPriority, -taskId])

    def rmv(self, taskId: int) -> None:

        del self.task_user_priority[taskId]

    def execTop(self) -> int:

        while self.task_priority:
            priority, task = heappop(self.task_priority)
            priority *= -1
            task *= -1

            if task not in self.task_user_priority or self.task_user_priority[task][1] != priority:
                continue

            userId = self.task_user_priority[task][0]
            del self.task_user_priority[task]
            return userId

        if len(self.task_priority) == 0:
            return -1


# Your TaskManager object will be instantiated and called as such:
# obj = TaskManager(tasks)
# obj.add(userId,taskId,priority)
# obj.edit(taskId,newPriority)
# obj.rmv(taskId)
# param_4 = obj.execTop()


# | 功能       | 时间复杂度               |
# | -------- | ------------------- |
# | `add`    | O(log N)            |
# | `update` | O(log N)            |
# | `remove` | O(1)                |
# | `exeTop` | 均摊 O(log N)，最坏 O(N) |

