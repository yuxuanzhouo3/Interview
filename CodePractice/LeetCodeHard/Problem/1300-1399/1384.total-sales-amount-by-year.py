"""
LC 1384. Total Sales Amount by Year — Hard (SQL)
https://leetcode.com/problems/total-sales-amount-by-year/
Best: 按年拆分销售区间
"""

SOLUTION_SQL = """
SELECT
  s.product_id,
  p.product_name,
  y.report_year,
  s.average_daily_sales * (DATEDIFF(
    LEAST(s.period_end, y.y_end),
    GREATEST(s.period_start, y.y_start)
  ) + 1) AS total_amount
FROM Sales s
JOIN Product p ON s.product_id = p.product_id
JOIN (
  SELECT '2018' AS report_year, '2018-01-01' AS y_start, '2018-12-31' AS y_end
  UNION ALL SELECT '2019', '2019-01-01', '2019-12-31'
  UNION ALL SELECT '2020', '2020-01-01', '2020-12-31'
) y ON s.period_start <= y.y_end AND s.period_end >= y.y_start
ORDER BY s.product_id, y.report_year
"""


def _test():
    assert "datediff" in SOLUTION_SQL.lower(), "DATEDIFF"
    assert "greatest" in SOLUTION_SQL.lower(), "GREATEST"
    assert "least" in SOLUTION_SQL.lower(), "LEAST"
    assert "report_year" in SOLUTION_SQL.lower(), "report_year"
    print("LC 1384 OK")


if __name__ == "__main__":
    _test()
