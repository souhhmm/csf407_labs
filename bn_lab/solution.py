from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
import random

SENTENCES = (
    "the cat sat on the mat", "the cat sat on the rug",
    "the dog sat on the mat", "the dog ran to the park",
    "the cat ran to the park", "the dog sat on the rug",
)
START, END = "<START>", "<END>"


class LanguageModel:
    def __init__(self, sentences, order=1):
        if order not in (1, 2):
            raise ValueError("Only orders 1 and 2 are supported")
        self.order = order
        self.counts = defaultdict(Counter)
        self.vocabulary = set()
        for sentence in sentences:
            words = sentence.lower().split()
            if not words or any(word in (START.lower(), END.lower()) for word in words):
                raise ValueError("Empty sentence or reserved token in training data")
            self.vocabulary.update(words)
            tokens = [START] * order + words + [END]
            for index in range(order, len(tokens)):
                self.counts[tuple(tokens[index - order:index])][tokens[index]] += 1

    def distribution(self, context):
        if isinstance(context, str):
            context = (context,)
        counts = self.counts.get(tuple(context), {})
        total = sum(counts.values())
        return {token: amount / total for token, amount in sorted(counts.items())} if total else {}

    def predict(self, context):
        probabilities = self.distribution(context)
        # alphabetical token order breaks ties reproducibly.
        return max(probabilities, key=probabilities.get) if probabilities else None

    def generate(self, rng, mode="sample", max_tokens=40):
        if mode not in ("sample", "greedy"):
            raise ValueError("Use sample or greedy generation")
        history = [START] * self.order
        words = []
        for _ in range(max_tokens):
            probabilities = self.distribution(history[-self.order:])
            if not probabilities:
                return {"text": " ".join(words), "stop": "unknown_context"}
            if mode == "greedy":
                token = self.predict(history[-self.order:])
            else:
                token = rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
            if token == END:
                return {"text": " ".join(words), "stop": "end"}
            words.append(token)
            history.append(token)
        return {"text": " ".join(words), "stop": "length_limit"}

    def validate(self):
        for context in self.counts:
            p = self.distribution(context)
            assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-12)
            assert all(0 < value <= 1 for value in p.values())

    def summary(self, samples):
        # these candidate tables include histories containing START outside the prefix.
        # such histories are structurally impossible, so this is a raw table-size measure.
        previous = sorted(self.vocabulary | {START})
        outcomes = self.vocabulary | {END}
        possible_contexts = list(product(previous, repeat=self.order))
        observed = len(self.counts)
        nonzero = sum(len(values) for values in self.counts.values())
        return {"candidate_contexts": len(possible_contexts), "observed_contexts": observed,
                "unseen_contexts": sum(context not in self.counts for context in possible_contexts),
                "nonzero_entries": nonzero, "free_parameters_on_observed_support": nonzero - observed,
                "dense_cpt_entries": len(possible_contexts) * len(outcomes),
                "dense_cpt_free_parameters": len(possible_contexts) * (len(outcomes) - 1),
                "zero_entries_on_observed_contexts": observed * len(outcomes) - nonzero,
                "unique_sentences_out_of_20": len({sample["text"] for sample in samples}),
                "terminated_out_of_20": sum(sample["stop"] == "end" for sample in samples)}


def main():
    models = {str(order): LanguageModel(SENTENCES, order) for order in (1, 2)}
    first = models["1"]
    # independently calculated counts for the handout's five contexts.
    expected = {
        "the": {"cat": 3, "dog": 3, "mat": 2, "rug": 2, "park": 2},
        "cat": {"sat": 2, "ran": 1}, "dog": {"sat": 2, "ran": 1},
        "sat": {"on": 4}, "ran": {"to": 2},
    }
    for context, counts in expected.items():
        assert first.counts[(context,)] == counts
    assert first.distribution("unseen") == {} and first.predict("unseen") is None
    assert models["2"].distribution(("the", "cat")) == {"ran": 1 / 3, "sat": 2 / 3}
    assert models["2"].distribution(("on", "the")) == {"mat": 0.5, "rug": 0.5}
    results = {}
    for order, model in models.items():
        model.validate()
        rng = random.Random(7)
        samples = [model.generate(rng) for _ in range(20)]
        greedy = [model.generate(rng, "greedy") for _ in range(5)]
        for sample in samples:
            history = [START] * model.order + sample["text"].split()
            if sample["stop"] == "end":
                history.append(END)
            for index in range(model.order, len(history)):
                assert history[index] in model.distribution(history[index - model.order:index])
        results[order] = {
            "summary": model.summary(samples),
            "counts": {" | ".join(context): dict(sorted(counts.items()))
                       for context, counts in sorted(model.counts.items())},
            "probabilities": {" | ".join(context): model.distribution(context)
                              for context in sorted(model.counts)},
            "predictions": {" | ".join(context): model.predict(context)
                            for context in sorted(model.counts)},
            "sampled_20": samples, "sampled_5": samples[:5], "greedy_5": greedy,
        }
    folder = Path(__file__).resolve().parent
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for order, result in results.items():
        lines = [f"{i:02d}. {sample['text']} [{sample['stop']}]"
                 for i, sample in enumerate(result["sampled_20"], 1)]
        (folder / f"generated_order{order}.txt").write_text("\n".join(lines) + "\n")
        print("Order", order, result["summary"])
        print("Greedy example:", result["greedy_5"][0])
    print("All count, normalisation, unknown-context, and generation checks passed.")


if __name__ == "__main__":
    main()
