import os
import unittest
from unittest.mock import patch

import gateway


class GatewayPolicyTests(unittest.TestCase):
    def test_local_coding_turn_does_not_attach_web_tools(self):
        request = {
            "model": "qwen-coder",
            "stream": True,
            "messages": [{"role": "user", "content": "fix this bug in gateway.py"}],
        }

        converted = gateway.anthropic_to_openai_request(request, "models/qwen")

        self.assertFalse(converted["_web_enabled_this_turn"])
        self.assertNotIn("tools", converted)

    def test_docs_turn_attaches_web_tools(self):
        request = {
            "model": "qwen-coder",
            "messages": [{"role": "user", "content": "check latest mlx-lm docs"}],
        }

        converted = gateway.anthropic_to_openai_request(request, "models/qwen")

        self.assertTrue(converted["_web_enabled_this_turn"])
        self.assertEqual(
            ["web_search", "web_fetch"],
            [tool["function"]["name"] for tool in converted["tools"]],
        )

    def test_user_can_disable_web_for_turn(self):
        request = {
            "model": "qwen-coder",
            "messages": [
                {"role": "user", "content": "do not search, who is Ashik Khan IIT KGP?"}
            ],
        }

        converted = gateway.anthropic_to_openai_request(request, "models/qwen")

        self.assertFalse(converted["_web_enabled_this_turn"])
        self.assertNotIn("tools", converted)

    def test_ask_policy_requires_explicit_web_intent(self):
        with patch.dict(os.environ, {"AUTO_WEB_VERIFY": "ask"}):
            self.assertFalse(gateway.request_needs_web_verification("who is Ada Lovelace?"))
            self.assertTrue(
                gateway.request_needs_web_verification("verify who Ada Lovelace is")
            )

    def test_search_query_cleanup(self):
        self.assertEqual(
            "latest mlx lm docs",
            gateway.search_query_for_request("please search for latest mlx lm docs online"),
        )


if __name__ == "__main__":
    unittest.main()
