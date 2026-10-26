import unittest
from src.actions import extract_actions


class ActionTests(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(extract_actions("회의 시작\n할 일:자료 조사"), ["자료 조사"])

    def test_empty(self):
        self.assertEqual(extract_actions(""), [])

    def test_whitespace_only(self):
        self.assertEqual(extract_actions(" \n\t"), [])

    def test_trim(self):
        self.assertEqual(extract_actions("  할 일:  발표 준비  "), ["발표 준비"])

    def test_blank_action(self):
        self.assertEqual(extract_actions("할 일:   "), [])

    def test_no_action(self):
        self.assertEqual(extract_actions("오늘은 일정만 논의했습니다."), [])

    def test_multiple(self):
        self.assertEqual(extract_actions("할 일:자료 조사\n결정:금요일 발표\n할 일:초안 작성"),
                         ["자료 조사", "초안 작성"])

    def test_colon_in_action(self):
        self.assertEqual(extract_actions("할 일:시간 확인:10시"), ["시간 확인:10시"])


if __name__ == "__main__":
    unittest.main()
