import string

class Tokenizer():
    def __init__(self):
        self.words_2_idx = {}
        self.idx_2_words = {}

    def fit(self, text: str):
        words = text.split()

        self.words_2_idx["<UNK>"] = 0

        for word in words:
            if word not in self.words_2_idx:
                self.words_2_idx[word] = len(self.words_2_idx)

        self.idx_2_words = {idx : word for word, idx in self.words_2_idx.items()}

    def encode(self, text: str) -> list:
        words = text.translate(str.maketrans('', '', string.punctuation)).split()
        indices = []
        for word in words:
            if word in self.words_2_idx:
                indices.append(self.words_2_idx[word])
            else:
                indices.append(self.words_2_idx["<UNK>"])
        return indices

    def decode(self, indices: list) -> str:
        words = [self.idx_2_words[idx] for idx in indices]
        return " ".join(words)


def main():
    t = Tokenizer()
    t.fit("the cat sat on the mat")

    print(t.encode("the cat! sat haha on the mat haha"))
    print(t.decode([1, 2, 3, 4, 1, 5, 0]))

if __name__ == "__main__":
    main()
