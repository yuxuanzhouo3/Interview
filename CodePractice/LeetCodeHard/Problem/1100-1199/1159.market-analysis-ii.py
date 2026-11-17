"""
LC 1159. Market Analysis II — Hard (SQL)
https://leetcode.com/problems/market-analysis-ii/
Best: 第二单是否卖家喜爱品牌
"""

SOLUTION_SQL = """
SELECT u.user_id AS seller_id,
       CASE WHEN i.item_brand = u.favorite_brand THEN 'yes' ELSE 'no' END AS 2nd_item_fav_brand
FROM Users u
LEFT JOIN (
  SELECT seller_id, item_id
  FROM (
    SELECT seller_id, item_id,
           ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY order_date) AS rn
    FROM Orders
  ) t
  WHERE rn = 2
) o ON u.user_id = o.seller_id
LEFT JOIN Items i ON o.item_id = i.item_id
"""


def _test():
    assert "row_number" in SOLUTION_SQL.lower(), "ROW_NUMBER"
    assert "favorite_brand" in SOLUTION_SQL.lower(), "favorite_brand"
    assert "rn = 2" in SOLUTION_SQL.lower(), "rn = 2"
    print("LC 1159 OK")


if __name__ == "__main__":
    _test()
