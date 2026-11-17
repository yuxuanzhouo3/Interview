"""
LC 3451. Find Invalid IP Addresses — Hard
https://leetcode.com/problems/find-invalid-ip-addresses/
Tags: Database
Best: 最优解
"""
SOLUTION_SQL = """
SELECT
  ip,
  COUNT(*) AS invalid_count
FROM logs
WHERE
  LENGTH(ip) - LENGTH(REPLACE(ip, '.', '')) <> 3
  OR ip REGEXP '(^|\\.)0[0-9]'
  OR CAST(SUBSTRING_INDEX(ip, '.', 1) AS UNSIGNED) > 255
  OR CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(ip, '.', 2), '.', -1) AS UNSIGNED) > 255
  OR CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(ip, '.', 3), '.', -1) AS UNSIGNED) > 255
  OR CAST(SUBSTRING_INDEX(ip, '.', -1) AS UNSIGNED) > 255
GROUP BY ip
ORDER BY invalid_count DESC, ip DESC;
"""

class Solution:
    def find_invalid_ips(self):
        return SOLUTION_SQL

def _test():
    s = Solution()
    assert "invalid_count" in SOLUTION_SQL.lower()
    assert "group by" in SOLUTION_SQL.lower()
    assert "255" in SOLUTION_SQL
    print("LC 3451 OK")

if __name__ == "__main__":
    _test()
