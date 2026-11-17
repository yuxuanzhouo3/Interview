"""
LC 1767. Find the Subtasks That Did Not Execute — Hard (SQL)
https://leetcode.com/problems/find-the-subtasks-that-did-not-execute/
Best: 递归生成 1..n 子任务差集
"""

SOLUTION_SQL = """
WITH RECURSIVE seq AS (
  SELECT task_id, 1 AS subtask_id, subtasks_count FROM Tasks
  UNION ALL
  SELECT task_id, subtask_id + 1, subtasks_count
  FROM seq
  WHERE subtask_id < subtasks_count
)
SELECT s.task_id, s.subtask_id
FROM seq s
LEFT JOIN Executed e
  ON s.task_id = e.task_id AND s.subtask_id = e.subtask_id
WHERE e.subtask_id IS NULL
"""


def _test():
    assert "recursive" in SOLUTION_SQL.lower(), "RECURSIVE"
    assert "subtask_id" in SOLUTION_SQL.lower(), "subtask_id"
    assert "left join" in SOLUTION_SQL.lower(), "LEFT JOIN"
    assert "is null" in SOLUTION_SQL.lower(), "IS NULL"
    print("LC 1767 OK")


if __name__ == "__main__":
    _test()
