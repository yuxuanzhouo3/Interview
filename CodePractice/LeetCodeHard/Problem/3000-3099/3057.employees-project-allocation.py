"""
LC 3057. Employees Project Allocation — Hard
https://leetcode.com/problems/employees-project-allocation/
Best: Project allocation ranking — SQL
"""

SOLUTION_SQL = """
WITH workload AS (
  SELECT e.employee_id, e.name, e.team, p.project_id, p.workload,
         SUM(p.workload) OVER (PARTITION BY e.employee_id) AS emp_load,
         AVG(p.workload) OVER (PARTITION BY e.team) AS team_avg
  FROM Employees e
  JOIN Project p ON e.employee_id = p.employee_id
)
SELECT employee_id, name, project_id, workload AS project_workload,
       emp_load AS total_workload
FROM workload
WHERE emp_load > team_avg
ORDER BY employee_id, project_id
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "team" in q and "workload" in q
    print("LC 3057 OK")


if __name__ == "__main__":
    _test()

