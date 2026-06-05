import torch
from tokenizer import Tokenizer

# Modular version of the embeddings.ipynb code, to be used in other notebooks

class WordEmbedder:
    def __init__(self, corpus: str, embedding_dim: int = 4):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit(corpus)
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.tokenizer.words_2_idx),
            embedding_dim=embedding_dim
        )

    def __call__(self, text: str):
        encoding = self.tokenizer.encode(text)
        encoding_tensor = torch.Tensor(encoding).int()
        return self.embedding(encoding_tensor)

def main():
    embedder = WordEmbedder("the cat sat on the mat")
    output = embedder("the cat! sat haha on the mat haha")
    print(output.shape)

if __name__ == "__main__":
    main()