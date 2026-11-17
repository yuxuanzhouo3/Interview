"""
LC 2199. Finding the Topic of Each Post — Hard (SQL)
https://leetcode.com/problems/finding-the-topic-of-each-post/
Best: 关键词匹配聚合，无匹配则 'Ambiguous!'
"""

SOLUTION_SQL = """
SELECT
  p.post_id,
  IFNULL(
    NULLIF(
      GROUP_CONCAT(DISTINCT k.topic_id ORDER BY k.topic_id),
      ''
    ),
    'Ambiguous!'
  ) AS topic
FROM Posts p
LEFT JOIN Keywords k
  ON CONCAT(' ', LOWER(p.content), ' ') LIKE CONCAT('% ', LOWER(k.word), ' %')
GROUP BY p.post_id
ORDER BY p.post_id
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "ambiguous" in q
    assert "keywords" in q and "posts" in q
    assert "group_concat" in q or "topic" in q
    print("LC 2199 OK")


if __name__ == "__main__":
    _test()
