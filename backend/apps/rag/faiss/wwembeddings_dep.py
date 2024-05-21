import requests
import json
import logging
import asyncio

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class WWEmbeddings:
    def __init__(self, model: str = "nomic-embed-text:latest", url: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.url = url

    def eraseNewline(self, text: str) -> str:
        return text.replace("\n", " ")

    def eraseNewlines(self, texts: list[str]) -> list[str]:
        return list(map(lambda x: x.replace("\n"," "), texts))

    def generate_ollama_embeddings(self, prompt: str):
        prompt = self.eraseNewline(prompt)
        data = {
            "model": self.model,
            "prompt": prompt
        }

        log.info(f"for embedding input=>[{prompt}]\n\n")
        response = requests.post(self.url, data=json.dumps(data), headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            log.info(f"input=>[{prompt}] : embeddings =>[{response.json()}]")
            return response.json()["embedding"]
        else:
            response.raise_for_status()

    def embed_documents(self, texts):
        return [self.generate_ollama_embeddings(text) for text in texts]

    def embed_query(self, text):
        return self.generate_ollama_embeddings(text)

    async def aembed_documents(self, texts):
        loop = asyncio.get_event_loop()
        return await asyncio.gather(*[loop.run_in_executor(None, self.generate_ollama_embeddings, text) for text in texts])

    async def aembed_query(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_ollama_embeddings, text)

    def __call__(self, text):
        return self.embed_query(text)
