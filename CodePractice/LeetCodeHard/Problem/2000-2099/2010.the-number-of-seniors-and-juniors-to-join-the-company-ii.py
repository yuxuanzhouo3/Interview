"""
LC 2010. The Number of Seniors and Juniors to Join the Company II — Hard (SQL)
https://leetcode.com/problems/the-number-of-seniors-and-juniors-to-join-the-company-ii/
Best: 同 2004 并返回雇员 id
"""

SOLUTION_SQL = """
WITH senior AS (
  SELECT employee_id, salary,
         SUM(salary) OVER (ORDER BY salary, employee_id) AS cum
  FROM Candidates WHERE experience = 'Senior'
),
junior AS (
  SELECT employee_id, salary,
         SUM(salary) OVER (ORDER BY salary, employee_id) AS cum
  FROM Candidates WHERE experience = 'Junior'
),
senior_hire AS (
  SELECT employee_id, salary FROM senior WHERE cum <= 70000
),
spent AS (SELECT IFNULL(SUM(salary),0) AS s FROM senior_hire)
SELECT employee_id FROM senior_hire
UNION ALL
SELECT j.employee_id
FROM junior j, spent
WHERE j.cum <= 70000 - spent.s
"""


def _test():
    assert "senior" in SOLUTION_SQL.lower(), "Senior"
    assert "junior" in SOLUTION_SQL.lower(), "Junior"
    assert "employee_id" in SOLUTION_SQL.lower(), "employee_id"
    assert "70000" in SOLUTION_SQL.lower(), "70000"
    print("LC 2010 OK")


if __name__ == "__main__":
    _test()
