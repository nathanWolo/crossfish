import sys
import unittest

from roundrobin import MatchBot, play_one


BOT_SCRIPT = r"""
import sys
import time

for raw in sys.stdin:
    parts = raw.split()
    if not parts:
        continue
    if parts[0] == "GO":
        time.sleep(int(parts[1]) / 1000.0)
        print("0 0", flush=True)
"""


class TestMatchBotDeadline(unittest.TestCase):
    def make_bot(self):
        return MatchBot(
            [sys.executable, "-u", "-c", BOT_SCRIPT], "deadline-test"
        )

    def test_timely_response(self):
        bot = self.make_bot()
        try:
            self.assertEqual(bot.go(1, timeout_ms=100), (0, 0))
            self.assertEqual(bot.timeouts, 0)
        finally:
            bot.close()

    def test_timeout_is_recorded_and_process_restarts(self):
        bot = self.make_bot()
        try:
            with self.assertRaises(TimeoutError):
                bot.go(50, timeout_ms=5)
            self.assertEqual(bot.timeouts, 1)
            self.assertEqual(bot.go(1, timeout_ms=100), (0, 0))
        finally:
            bot.close()


class TimeoutBot:
    def new(self):
        pass

    def apply(self, mb, sq):
        pass

    def go(self, ms):
        raise TimeoutError("synthetic timeout")


class TestTimeoutScoring(unittest.TestCase):
    def test_b1_timeout_is_a_loss(self):
        self.assertEqual(
            play_one(TimeoutBot(), TimeoutBot(), [], 20, True), -1
        )

    def test_b2_timeout_is_a_b1_win(self):
        self.assertEqual(
            play_one(TimeoutBot(), TimeoutBot(), [], 20, False), 1
        )


if __name__ == "__main__":
    unittest.main()
