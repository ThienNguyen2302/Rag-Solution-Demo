import unittest

from helpers import parse_step_back_question


class StepBackTests(unittest.TestCase):
    def test_parse_step_back_question_strips_model_formatting(self):
        response = "```text\nWhat broader principles explain this idea?\n```"

        self.assertEqual(
            parse_step_back_question(response),
            "What broader principles explain this idea?",
        )

    def test_parse_step_back_question_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_step_back_question("   ")


if __name__ == "__main__":
    unittest.main()
