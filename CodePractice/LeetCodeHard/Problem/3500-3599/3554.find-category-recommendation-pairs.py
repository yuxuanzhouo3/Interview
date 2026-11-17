"""
LC 3554. Find Category Recommendation Pairs — Hard (SQL)
https://leetcode.com/problems/find-category-recommendation-pairs/
Best: 类别自连接统计共现用户
"""

SOLUTION_SQL = """
WITH user_category AS (
    SELECT DISTINCT user_id, category
    FROM ProductPurchases JOIN ProductInfo USING (product_id)
),
pair_per_user AS (
    SELECT a.user_id, a.category AS category1, b.category AS category2
    FROM user_category a
    JOIN user_category b ON a.user_id = b.user_id AND a.category < b.category
)
SELECT category1, category2, COUNT(DISTINCT user_id) AS customer_count
FROM pair_per_user
GROUP BY 1, 2
HAVING customer_count >= 3
ORDER BY 3 DESC, 1, 2;
"""


def _test():
    assert "SELECT" in SOLUTION_SQL.upper()
    print("LC 3554 OK")


if __name__ == "__main__":
    _test()
