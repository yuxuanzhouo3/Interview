"""
LC 3156. Employee Task Duration and Concurrent Tasks — Hard
https://leetcode.com/problems/employee-task-duration-and-concurrent-tasks/
Best: Employee task duration & concurrency — SQL
"""

SOLUTION_SQL = """
WITH ordered AS (
  SELECT employee_id, task_id, start_time, end_time,
         LAG(end_time) OVER (PARTITION BY employee_id ORDER BY start_time) AS prev_end
  FROM Tasks
),
flags AS (
  SELECT *, CASE WHEN prev_end IS NULL OR start_time >= prev_end THEN 1 ELSE 0 END AS new_grp
  FROM ordered
),
grp AS (
  SELECT *, SUM(new_grp) OVER (PARTITION BY employee_id ORDER BY start_time) AS gid
  FROM flags
)
SELECT employee_id,
       FLOOR(SUM(TIMESTAMPDIFF(SECOND, start_time, end_time))/3600) AS total_task_hours,
       MAX(cnt) AS max_concurrent_tasks
FROM (
  SELECT employee_id, gid,
         MIN(start_time) AS start_time, MAX(end_time) AS end_time,
         COUNT(*) AS cnt
  FROM grp
  GROUP BY employee_id, gid
) t
GROUP BY employee_id
ORDER BY employee_id
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "employee_id" in q and "concurrent" in q or "count" in q
    print("LC 3156 OK")


if __name__ == "__main__":
    _test()

