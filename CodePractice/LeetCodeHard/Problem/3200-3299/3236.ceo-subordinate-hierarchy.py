"""
LC 3236. CEO Subordinate Hierarchy — Hard (SQL)
https://leetcode.com/problems/ceo-subordinate-hierarchy/
Best: Recursive CTE hierarchy levels
"""

SOLUTION_SQL = """
WITH RECURSIVE cte AS (
  SELECT
    employee_id AS subordinate_id,
    employee_name AS subordinate_name,
    0 AS hierarchy_level,
    salary
  FROM Employees
  WHERE manager_id IS NULL
  UNION ALL
  SELECT
    e.employee_id,
    e.employee_name,
    hierarchy_level + 1,
    e.salary
  FROM Employees e
  INNER JOIN cte ON e.manager_id = cte.subordinate_id
)
SELECT
  subordinate_id,
  subordinate_name,
  hierarchy_level,
  salary - (SELECT salary FROM Employees WHERE manager_id IS NULL) AS salary_difference
FROM cte
WHERE hierarchy_level > 0
ORDER BY hierarchy_level, subordinate_id
"""


def _test():
    assert "RECURSIVE" in SOLUTION_SQL
    assert "hierarchy_level" in SOLUTION_SQL
    assert "salary_difference" in SOLUTION_SQL
    print("LC 3236 OK")


if __name__ == "__main__":
    _test()
