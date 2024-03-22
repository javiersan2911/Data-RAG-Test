import unittest

from openai import OpenAI


class TestOpenAI(unittest.TestCase):
    def test_open_ai_key(self):
        client = OpenAI(api_key="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")

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
            ]
        )
        self.assertTrue(len(completion.choices) > 0)  # add assertion here


if __name__ == '__main__':
    unittest.main()
