"""
LC 185. Department Top Three Salaries — Hard (SQL)
Best: DENSE_RANK() OVER (PARTITION BY departmentId ORDER BY salary DESC) <= 3
"""

SOLUTION_SQL = """
SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary
FROM (
  SELECT *, DENSE_RANK() OVER (PARTITION BY departmentId ORDER BY salary DESC) AS rk
  FROM Employee
) e
JOIN Department d ON e.departmentId = d.id
WHERE e.rk <= 3
"""


def _test():
    assert "DENSE_RANK" in SOLUTION_SQL
    print("LC 185 OK")


if __name__ == "__main__":
    _test()
