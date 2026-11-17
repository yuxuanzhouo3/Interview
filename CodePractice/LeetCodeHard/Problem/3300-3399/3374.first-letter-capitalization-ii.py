"""
LC 3374. First Letter Capitalization II — Hard (SQL / Pandas)
https://leetcode.com/problems/first-letter-capitalization-ii/
Best: Split words/hyphen parts + capitalize
"""

SOLUTION_SQL = """
-- MySQL 8+: recursive/word split awkward; prefer application/pandas style.
-- Conceptual: split content_text by space; for each token split by '-',
-- INITCAP each part, rejoin with '-' then spaces.
SELECT content_id,
       content_text AS original_text,
       content_text AS converted_text  -- placeholder; use pandas convert below
FROM user_content
"""


def convert_text(text: str) -> str:
    return " ".join(
        "-".join(part.capitalize() for part in word.split("-"))
        if "-" in word
        else word.capitalize()
        for word in text.split(" ")
    )


def capitalize_content(rows):
    """rows: list of (content_id, content_text) -> list of dicts"""
    out = []
    for cid, text in rows:
        out.append(
            {
                "content_id": cid,
                "original_text": text,
                "converted_text": convert_text(text),
            }
        )
    return out


def _test():
    rows = [
        (1, "hello world of SQL"),
        (2, "the QUICK-brown fox"),
        (3, "modern-day DATA science"),
        (4, "web-based FRONT-end development"),
    ]
    ans = capitalize_content(rows)
    assert ans[0]["converted_text"] == "Hello World Of Sql"
    assert ans[1]["converted_text"] == "The Quick-Brown Fox"
    assert ans[2]["converted_text"] == "Modern-Day Data Science"
    assert ans[3]["converted_text"] == "Web-Based Front-End Development"
    print("LC 3374 OK")


if __name__ == "__main__":
    _test()
