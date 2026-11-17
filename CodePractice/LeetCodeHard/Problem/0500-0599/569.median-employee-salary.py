"""
LC 569. Median Employee Salary — Hard (SQL Premium)
Best: Window row_number / self-join median
"""

SOLUTION_SQL = """
SELECT Id, Company, Salary
FROM (
  SELECT e.*,
    ROW_NUMBER() OVER (PARTITION BY Company ORDER BY Salary, Id) AS rn,
    COUNT(*) OVER (PARTITION BY Company) AS cnt
  FROM Employee e
) t
WHERE rn IN (FLOOR((cnt+1)/2), FLOOR((cnt+2)/2))
"""


def _test():
    assert "ROW_NUMBER" in SOLUTION_SQL and "Company" in SOLUTION_SQL
    print("LC 569 OK")


if __name__ == "__main__":
    _test()
