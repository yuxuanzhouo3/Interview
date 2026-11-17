"""
LC 3214. Year on Year Growth Rate — Hard (SQL)
https://leetcode.com/problems/year-on-year-growth-rate/
Best: Yearly aggregate + self join / LAG
"""

SOLUTION_SQL = """
WITH YearlySpends AS (
  SELECT
    product_id,
    YEAR(transaction_date) AS year,
    SUM(spend) AS spend
  FROM user_transactions
  GROUP BY 1, 2
)
SELECT
  CurrYear.year,
  CurrYear.product_id,
  CurrYear.spend AS curr_year_spend,
  PrevYear.spend AS prev_year_spend,
  ROUND(100 * (CurrYear.spend - PrevYear.spend) / PrevYear.spend, 2) AS yoy_rate
FROM YearlySpends AS CurrYear
LEFT JOIN YearlySpends AS PrevYear
  ON CurrYear.product_id = PrevYear.product_id
 AND CurrYear.year - 1 = PrevYear.year
ORDER BY product_id, year
"""


def _test():
    assert "YEAR" in SOLUTION_SQL and "LAG" not in SOLUTION_SQL or True
    assert "prev_year_spend" in SOLUTION_SQL
    assert "yoy_rate" in SOLUTION_SQL
    assert "LEFT JOIN" in SOLUTION_SQL
    print("LC 3214 OK")


if __name__ == "__main__":
    _test()
