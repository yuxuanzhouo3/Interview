"""
LC 1225. Report Contiguous Dates — Hard (SQL)
https://leetcode.com/problems/report-contiguous-dates/
Best: 成功/失败日期岛起止
"""

SOLUTION_SQL = """
WITH days AS (
  SELECT fail_date AS dt, 'failed' AS period_state FROM Failed
  WHERE fail_date BETWEEN '2019-01-01' AND '2019-12-31'
  UNION ALL
  SELECT success_date, 'succeeded' FROM Succeeded
  WHERE success_date BETWEEN '2019-01-01' AND '2019-12-31'
),
marked AS (
  SELECT dt, period_state,
         ROW_NUMBER() OVER (ORDER BY dt)
         - ROW_NUMBER() OVER (PARTITION BY period_state ORDER BY dt) AS grp
  FROM days
)
SELECT period_state, MIN(dt) AS start_date, MAX(dt) AS end_date
FROM marked
GROUP BY period_state, grp
ORDER BY start_date
"""


def _test():
    assert "row_number" in SOLUTION_SQL.lower(), "ROW_NUMBER"
    assert "union all" in SOLUTION_SQL.lower(), "UNION ALL"
    assert "period_state" in SOLUTION_SQL.lower(), "period_state"
    assert "grp" in SOLUTION_SQL.lower(), "grp"
    print("LC 1225 OK")


if __name__ == "__main__":
    _test()
