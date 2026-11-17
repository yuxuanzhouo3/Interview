"""
LC 3368. First Letter Capitalization — Hard (SQL)
https://leetcode.com/problems/first-letter-capitalization/
Best: Recursive CTE split words + capitalize
"""

SOLUTION_SQL = """
WITH RECURSIVE Words AS (
  SELECT
    content_id,
    SUBSTRING_INDEX(content_text, ' ', 1) AS word,
    SUBSTRING(content_text, LENGTH(SUBSTRING_INDEX(content_text, ' ', 1)) + 2) AS remaining_text,
    1 AS token_index
  FROM user_content
  UNION ALL
  SELECT
    content_id,
    SUBSTRING_INDEX(remaining_text, ' ', 1),
    SUBSTRING(remaining_text, LENGTH(SUBSTRING_INDEX(remaining_text, ' ', 1)) + 2),
    token_index + 1
  FROM Words
  WHERE remaining_text != ''
),
Converted AS (
  SELECT
    content_id,
    GROUP_CONCAT(
      CONCAT(UPPER(SUBSTRING(word, 1, 1)), LOWER(SUBSTRING(word, 2)))
      ORDER BY token_index SEPARATOR ' '
    ) AS converted_text
  FROM Words
  GROUP BY 1
)
SELECT
  u.content_id,
  u.content_text AS original_text,
  c.converted_text
FROM user_content u
JOIN Converted c USING (content_id)
"""


def convert_text(text: str) -> str:
    return " ".join(w.capitalize() for w in text.split(" "))


def _test():
    assert convert_text("hello WORLD") == "Hello World"
    assert "RECURSIVE" in SOLUTION_SQL
    assert "converted_text" in SOLUTION_SQL
    print("LC 3368 OK")


if __name__ == "__main__":
    _test()
