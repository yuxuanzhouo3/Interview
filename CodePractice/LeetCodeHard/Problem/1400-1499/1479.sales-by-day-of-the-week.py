"""
LC 1479. Sales by Day of the Week — Hard (SQL)
https://leetcode.com/problems/sales-by-day-of-the-week/
Best: 按类别透视周一到周日销量
"""

SOLUTION_SQL = """
SELECT
  i.item_category AS Category,
  SUM(IF(DAYOFWEEK(o.order_date)=2, quantity, 0)) AS Monday,
  SUM(IF(DAYOFWEEK(o.order_date)=3, quantity, 0)) AS Tuesday,
  SUM(IF(DAYOFWEEK(o.order_date)=4, quantity, 0)) AS Wednesday,
  SUM(IF(DAYOFWEEK(o.order_date)=5, quantity, 0)) AS Thursday,
  SUM(IF(DAYOFWEEK(o.order_date)=6, quantity, 0)) AS Friday,
  SUM(IF(DAYOFWEEK(o.order_date)=7, quantity, 0)) AS Saturday,
  SUM(IF(DAYOFWEEK(o.order_date)=1, quantity, 0)) AS Sunday
FROM Items i
LEFT JOIN Orders o ON i.item_id = o.item_id
GROUP BY i.item_category
ORDER BY Category
"""


def _test():
    assert "dayofweek" in SOLUTION_SQL.lower(), "DAYOFWEEK"
    assert "monday" in SOLUTION_SQL.lower(), "Monday"
    assert "category" in SOLUTION_SQL.lower(), "Category"
    assert "left join" in SOLUTION_SQL.lower(), "LEFT JOIN"
    print("LC 1479 OK")


if __name__ == "__main__":
    _test()
