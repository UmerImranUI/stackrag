import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


class Generator:
    """
    Local LLM generator using Ollama.

    Runs entirely on the user's machine.
    No paid API is required.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ollama_url: str = OLLAMA_URL,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate an answer from a prompt using Ollama.
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=500,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response"
            )

        return answer


if __name__ == "__main__":
    generator = Generator()

    test_prompt = """
You are a technical assistant.

Answer the following question in two concise sentences.

Question:
What is Apache Spark?
"""

    print("=" * 80)
    print("OLLAMA GENERATOR TEST")
    print("=" * 80)

    answer = generator.generate(test_prompt)

    print("\nANSWER:")
    print(answer)