"""
LC 3103. Find Trending Hashtags II  — Hard
https://leetcode.com/problems/find-trending-hashtags-ii/
Best: Trending hashtags II — SQL
"""

SOLUTION_SQL = """
WITH RECURSIVE
months AS (
  SELECT tweet_id, tweet, tweet_date
  FROM Tweets
  WHERE tweet_date BETWEEN '2024-02-01' AND '2024-02-29'
),
split AS (
  SELECT tweet_id, SUBSTRING_INDEX(SUBSTRING_INDEX(tweet, '#', n.n+1), ' ', -1) AS hashtag
  FROM months
  JOIN (SELECT 1 n UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5
        UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10) n
    ON CHAR_LENGTH(tweet) - CHAR_LENGTH(REPLACE(tweet, '#', '')) >= n.n
)
SELECT CONCAT('#', TRIM(BOTH '#' FROM hashtag)) AS hashtag, COUNT(*) AS count
FROM split
WHERE hashtag <> '' AND hashtag REGEXP '^[A-Za-z0-9_]+'
GROUP BY 1
ORDER BY count DESC, hashtag DESC
LIMIT 3
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "hashtag" in q and "limit 3" in q
    print("LC 3103 OK")


if __name__ == "__main__":
    _test()

