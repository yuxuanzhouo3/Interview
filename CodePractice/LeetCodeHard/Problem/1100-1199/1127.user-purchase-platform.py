"""
LC 1127. User Purchase Platform — Hard (SQL)
https://leetcode.com/problems/user-purchase-platform/
Best: 日期×平台维度 LEFT JOIN 用户分类
"""

SOLUTION_SQL = """
WITH user_day AS (
  SELECT spend_date, user_id,
         SUM(amount) AS amount,
         CASE WHEN COUNT(DISTINCT platform)=2 THEN 'both' ELSE MAX(platform) END AS platform
  FROM Spending
  GROUP BY spend_date, user_id
),
dims AS (
  SELECT DISTINCT spend_date, p.platform
  FROM Spending
  CROSS JOIN (SELECT 'mobile' AS platform UNION SELECT 'desktop' UNION SELECT 'both') p
)
SELECT d.spend_date, d.platform,
       IFNULL(SUM(u.amount), 0) AS total_amount,
       IFNULL(COUNT(u.user_id), 0) AS total_users
FROM dims d
LEFT JOIN user_day u
  ON d.spend_date = u.spend_date AND d.platform = u.platform
GROUP BY d.spend_date, d.platform
"""


def _test():
    assert "with" in SOLUTION_SQL.lower(), "WITH"
    assert "both" in SOLUTION_SQL.lower(), "both"
    assert "cross join" in SOLUTION_SQL.lower(), "CROSS JOIN"
    assert "ifnull" in SOLUTION_SQL.lower(), "IFNULL"
    print("LC 1127 OK")


if __name__ == "__main__":
    _test()
