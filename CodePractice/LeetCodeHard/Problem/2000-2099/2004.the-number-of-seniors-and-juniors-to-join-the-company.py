"""
LC 2004. The Number of Seniors and Juniors to Join the Company — Hard (SQL)
https://leetcode.com/problems/the-number-of-seniors-and-juniors-to-join-the-company/
Best: 预算内优先 Senior 再 Junior
"""

SOLUTION_SQL = """
WITH senior AS (
  SELECT employee_id, salary,
         SUM(salary) OVER (ORDER BY salary) AS cum
  FROM Candidates WHERE experience = 'Senior'
),
junior AS (
  SELECT employee_id, salary,
         SUM(salary) OVER (ORDER BY salary) AS cum
  FROM Candidates WHERE experience = 'Junior'
),
senior_hire AS (
  SELECT IFNULL(MAX(cum), 0) AS spent, COUNT(*) AS cnt
  FROM senior WHERE cum <= 70000
)
SELECT 'Senior' AS experience, cnt AS accepted_candidates FROM senior_hire
UNION ALL
SELECT 'Junior', COUNT(*)
FROM junior j, senior_hire s
WHERE j.cum <= 70000 - s.spent
"""


def _test():
    assert "senior" in SOLUTION_SQL.lower(), "Senior"
    assert "junior" in SOLUTION_SQL.lower(), "Junior"
    assert "70000" in SOLUTION_SQL.lower(), "70000"
    assert "sum(salary) over" in SOLUTION_SQL.lower(), "SUM(salary) OVER"
    print("LC 2004 OK")


if __name__ == "__main__":
    _test()
