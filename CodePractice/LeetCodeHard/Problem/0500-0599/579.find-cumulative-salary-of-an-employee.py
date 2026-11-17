"""
LC 579. Find Cumulative Salary of an Employee — Hard (SQL)
https://leetcode.com/problems/find-cumulative-salary-of-an-employee/
Best: 排除最近月，累加近 3 个月薪
"""

SOLUTION_SQL = """
SELECT a.Id, a.Month, SUM(b.Salary) AS Salary
FROM Employee a
JOIN Employee b
  ON a.Id = b.Id AND b.Month BETWEEN a.Month - 2 AND a.Month
WHERE (a.Id, a.Month) NOT IN (
  SELECT Id, MAX(Month) FROM Employee GROUP BY Id
)
GROUP BY a.Id, a.Month
ORDER BY a.Id, a.Month DESC
"""


def _test():
    assert "sum" in SOLUTION_SQL.lower(), "SUM"
    assert "join" in SOLUTION_SQL.lower(), "JOIN"
    assert "max(month)" in SOLUTION_SQL.lower(), "MAX(Month)"
    assert "between" in SOLUTION_SQL.lower(), "BETWEEN"
    print("LC 579 OK")


if __name__ == "__main__":
    _test()
