"""Tests for mbpp_scoring.py. These run real (tiny) Python subprocesses.

Run: python reward_functions/test_mbpp_scoring.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbpp_scoring as s  # noqa: E402

TESTS = ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 0) == 0"]


def gt(tests=TESTS, setup=""):
    return json.dumps({"tests": tests, "setup": setup})


def fence(code):
    return f"Here you go:\n```python\n{code}\n```\n"


class TestExtraction(unittest.TestCase):
    def test_first_python_block_wins(self):
        self.assertEqual(s.extract_code("a ```python\nx = 1\n``` b ```python\ny = 2\n```"), "x = 1")

    def test_no_python_fence_means_no_code(self):
        self.assertIsNone(s.extract_code("just prose"))
        self.assertIsNone(s.extract_code("```\nx = 1\n```"))  # untagged fence

    def test_code_inside_thinking_is_not_the_answer(self):
        r = "<think>try ```python\nx = 1\n``` maybe</think>\nAnswer:\n```python\nx = 2\n```"
        self.assertEqual(s.extract_code(r), "x = 2")

    def test_thinking_without_a_final_answer_has_no_code(self):
        self.assertIsNone(s.extract_code("<think>hmm ```python\nx = 1\n``` still going"))  # cut off
        self.assertIsNone(s.extract_code("<think>reasoning</think> no code here"))

    def test_closing_tag_alone_is_enough(self):
        """Some Qwen3 templates put the opening <think> in the prompt."""
        self.assertEqual(s.extract_code("reasoning ```python\nx = 1\n``` more</think>\n```python\nx = 2\n```"), "x = 2")


class TestScoring(unittest.TestCase):
    def test_all_tests_pass(self):
        self.assertEqual(s.score(fence("def add(a, b):\n    return a + b"), gt()), 1.0)

    def test_partial_credit_is_the_fraction_passed(self):
        # right for (1,2) and (2,2) only by coincidence of 'a + 1', wrong for (0,0)
        code = "def add(a, b):\n    return 3 if a == 1 else (4 if a == 2 else 99)"
        self.assertAlmostEqual(s.score(fence(code), gt()), 2 / 3)

    def test_no_code_block_is_the_format_penalty(self):
        self.assertEqual(s.score("I think the answer is 3", gt()), -0.25)

    def test_wrong_program_scores_zero_not_the_penalty(self):
        self.assertEqual(s.score(fence("def add(a, b):\n    return 0"), gt(["assert add(1, 2) == 3"])), 0.0)

    def test_crash_or_syntax_error_scores_zero(self):
        self.assertEqual(s.score(fence("def add(a, b:\n    return"), gt()), 0.0)
        self.assertEqual(s.score(fence("raise RuntimeError('boom')"), gt()), 0.0)
        self.assertEqual(s.score(fence("import sys\nsys.exit(0)"), gt()), 0.0)

    def test_infinite_loop_times_out_with_zero(self):
        self.assertEqual(s.score(fence("while True:\n    pass"), gt(), timeout=1.5), 0.0)

    def test_setup_code_is_available_to_the_tests(self):
        spec = gt(["assert helper() == 7"], setup="def helper():\n    return 7")
        self.assertEqual(s.score(fence("x = 1"), spec), 1.0)

    def test_multiline_and_failing_assert_handled_per_test(self):
        tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 99"]
        self.assertEqual(s.score(fence("def add(a, b):\n    return a + b"), gt(tests)), 0.5)


class TestCannotBeGamed(unittest.TestCase):
    def test_printing_fake_pass_markers_earns_nothing(self):
        code = "for i in range(3):\n    print(f'__R0_{i}_PASS')\n    print(f'__RDEADBEEF_{i}_PASS')\ndef add(a, b):\n    return 0"
        self.assertEqual(s.score(fence(code), gt()), 1 / 3)  # only add(0, 0) == 0 genuinely holds

    def test_generated_code_cannot_see_this_process_environment(self):
        os.environ["SECRET_FOR_TEST"] = "hunter2"
        try:
            code = "import os\ndef add(a, b):\n    return os.environ.get('SECRET_FOR_TEST')"
            self.assertEqual(s.score(fence(code), gt(["assert add(1, 2) is None"])), 1.0)
        finally:
            del os.environ["SECRET_FOR_TEST"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
