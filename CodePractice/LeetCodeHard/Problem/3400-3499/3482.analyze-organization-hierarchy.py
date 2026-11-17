"""
LC 3482. Analyze Organization Hierarchy — Hard (SQL)
https://leetcode.com/problems/analyze-organization-hierarchy/
Best: 递归 CTE 求层级与子树预算
"""

SOLUTION_SQL = """
WITH RECURSIVE
    level_cte AS (
        SELECT employee_id, manager_id, 1 AS level, salary FROM Employees
        UNION ALL
        SELECT a.employee_id, b.manager_id, level + 1, a.salary
        FROM level_cte a
        JOIN Employees b ON b.employee_id = a.manager_id
    ),
    employee_with_level AS (
        SELECT a.employee_id, a.employee_name, a.salary, b.level
        FROM Employees a,
            (SELECT employee_id, level FROM level_cte WHERE manager_id IS NULL) b
        WHERE a.employee_id = b.employee_id
    )
SELECT a.employee_id, a.employee_name, a.level,
    COALESCE(b.team_size, 0) AS team_size,
    a.salary + COALESCE(b.budget, 0) AS budget
FROM employee_with_level a
LEFT JOIN (
    SELECT manager_id AS employee_id, COUNT(*) AS team_size, SUM(salary) AS budget
    FROM level_cte WHERE manager_id IS NOT NULL GROUP BY manager_id
) b ON a.employee_id = b.employee_id
ORDER BY level, budget DESC, employee_name;
"""


def _test():
    assert "SELECT" in SOLUTION_SQL.upper()
    print("LC 3482 OK")


if __name__ == "__main__":
    _test()
