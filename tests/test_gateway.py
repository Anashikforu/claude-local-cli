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


class GatewayConversionTests(unittest.TestCase):
    def test_qwen_special_tokens_are_removed_without_breaking_stream_spaces(self):
        self.assertEqual(
            "Hello",
            gateway.clean_model_text(" Hello<|im_end|>"),
        )
        self.assertEqual(
            " How",
            gateway.clean_model_text(" How<|im_end|>", strip=False),
        )

    def test_anthropic_tool_use_converts_to_openai_tool_call(self):
        converted = gateway.convert_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "web_search",
                            "input": {"query": "mlx"},
                        }
                    ],
                }
            ]
        )

        self.assertEqual("assistant", converted[0]["role"])
        self.assertEqual("web_search", converted[0]["tool_calls"][0]["function"]["name"])

    def test_tool_result_converts_to_openai_tool_message(self):
        converted = gateway.convert_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "result text",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            [{"role": "tool", "tool_call_id": "toolu_1", "content": "result text"}],
            converted,
        )

    def test_openai_response_converts_to_anthropic_message(self):
        response = gateway.openai_to_anthropic_response(
            {
                "id": "chatcmpl_1",
                "choices": [
                    {
                        "message": {
                            "content": "Hello<|im_end|>",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "web_fetch",
                                        "arguments": '{"url":"https://example.com"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            "qwen-coder",
            {"messages": []},
        )

        self.assertEqual("Hello", response["content"][0]["text"])
        self.assertEqual("tool_use", response["content"][1]["type"])
        self.assertEqual("web_fetch", response["content"][1]["name"])

    def test_plain_json_tool_call_detection(self):
        tool_call = gateway.extract_plain_text_tool_call(
            """```json
            {"name":"web_search","arguments":{"query":"Ashik Khan IIT KGP"}}
            ```"""
        )

        self.assertEqual(
            {"name": "web_search", "arguments": {"query": "Ashik Khan IIT KGP"}},
            tool_call,
        )

    def test_model_aliases_resolve_to_default_model(self):
        self.assertEqual(
            "models/qwen",
            gateway.resolve_model("claude-sonnet-4-5", "models/qwen"),
        )
        self.assertTrue(gateway.is_available_model("qwen-coder", "models/qwen"))


class GatewayWebPolicyTests(unittest.TestCase):
    def test_all_auto_web_policies(self):
        with patch.dict(os.environ, {"WEB_POLICY": "off"}):
            self.assertFalse(gateway.request_needs_web_verification("latest mlx docs"))

        with patch.dict(os.environ, {"WEB_POLICY": "force"}):
            self.assertTrue(gateway.request_needs_web_verification("fix this bug"))

        with patch.dict(os.environ, {"WEB_POLICY": "ask"}):
            self.assertFalse(gateway.request_needs_web_verification("who is Ada Lovelace?"))
            self.assertTrue(gateway.request_needs_web_verification("search Ada Lovelace"))

    def test_search_depth_and_context_size_aliases(self):
        with patch.dict(os.environ, {"SEARCH_DEPTH": "quick"}, clear=False):
            self.assertEqual("fast", gateway.auto_web_mode())
            self.assertEqual("low", gateway.web_context_size())
            self.assertEqual(0, gateway.auto_web_fetch_count())

        with patch.dict(os.environ, {"SEARCH_DEPTH": "agentic"}, clear=False):
            self.assertEqual("balanced", gateway.auto_web_mode())
            self.assertEqual("medium", gateway.web_context_size())
            self.assertEqual(1, gateway.auto_web_fetch_count())

    def test_domain_filters(self):
        search_result = {
            "provider": "test",
            "query": "docs",
            "results": [
                {"title": "Python", "url": "https://docs.python.org/3/", "snippet": ""},
                {"title": "Reddit", "url": "https://reddit.com/r/python", "snippet": ""},
            ],
        }

        with patch.dict(
            os.environ,
            {
                "WEB_ALLOWED_DOMAINS": "docs.python.org",
                "WEB_BLOCKED_DOMAINS": "reddit.com",
            },
        ):
            filtered = gateway.apply_domain_filters(search_result)

        self.assertEqual(
            ["https://docs.python.org/3/"],
            [result["url"] for result in filtered["results"]],
        )

    def test_fetch_priority_prefers_official_academic_sources(self):
        self.assertLess(
            gateway.fetch_priority({"url": "https://cse.iitkgp.ac.in/~x/"}),
            gateway.fetch_priority({"url": "https://www.linkedin.com/in/x"}),
        )

    def test_tool_error_is_returned_as_json(self):
        result = gateway.execute_local_tool("web_fetch", {"url": "ftp://bad"}, 1)
        self.assertIn("web_fetch requires an http or https URL", result)


if __name__ == "__main__":
    unittest.main()
