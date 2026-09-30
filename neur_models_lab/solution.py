import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/csf407-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/csf407-cache")
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

STEPS, LEARNING_RATE, SEED = 5000, 0.03, 0
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
Y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float64)
THREE_CLASS = torch.tensor([0, 1, 1, 2])


class Network(nn.Module):
    def __init__(self, activation="tanh", outputs=1):
        super().__init__()
        self.hidden = nn.Linear(2, 2, dtype=torch.float64)
        self.activation = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]()
        self.output = nn.Linear(2, outputs, dtype=torch.float64)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))


def train(activation="tanh", zero_weights=False, outputs=1, seed=SEED):
    torch.manual_seed(seed)
    model = Network(activation, outputs)
    if zero_weights:
        with torch.no_grad():
            model.hidden.weight.zero_()
            model.output.weight.zero_()
        # the handout specifies zero weights, not zero biases; retain random biases.
    target = Y if outputs == 1 else THREE_CLASS
    loss_fn = nn.BCEWithLogitsLoss() if outputs == 1 else nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    initial = float(loss_fn(model(X), target).detach())
    history, symmetry = [], []
    early_gradient = None
    for step in range(STEPS):
        optimiser.zero_grad()
        logits = model(X)  # forward pass.
        loss = loss_fn(logits, target)  # mean over the four examples.
        loss.backward()  # reverse-mode automatic differentiation.
        if step == 0:
            early_gradient = model.hidden.weight.grad.detach().clone()
        if step in (0, 1, 10, 100, 1000, STEPS - 1):
            symmetry.append({"step": step, "weights": model.hidden.weight.detach().tolist(),
                             "biases": model.hidden.bias.detach().tolist(),
                             "row_difference": float(torch.linalg.vector_norm(
                                 model.hidden.weight[0] - model.hidden.weight[1]).detach())})
        history.append(float(loss.detach()))
        optimiser.step()  # apply the parameter update.
    with torch.no_grad():
        logits = model(X)
        final = float(loss_fn(logits, target))
        probabilities = torch.sigmoid(logits) if outputs == 1 else torch.softmax(logits, dim=1)
        predictions = (probabilities >= 0.5).long().flatten() if outputs == 1 else probabilities.argmax(1)
    correct = bool(torch.equal(predictions, target.long().flatten()))
    result = {"initial_loss": initial, "final_loss": final,
              "probabilities": probabilities.tolist(), "predictions": predictions.tolist(),
              "all_correct": correct, "early_gradient": early_gradient.tolist(),
              "early_gradient_norm": float(torch.linalg.vector_norm(early_gradient)),
              "hidden_preactivations": model.hidden(X).detach().tolist(),
              "hidden_activations": model.activation(model.hidden(X)).detach().tolist(),
              "hidden_weights": model.hidden.weight.detach().tolist(),
              "hidden_biases": model.hidden.bias.detach().tolist(), "snapshots": symmetry}
    return model, result, history


def check_gradient(model):
    criterion = nn.BCEWithLogitsLoss()
    model.zero_grad()
    criterion(model(X), Y).backward()
    gradient = model.hidden.weight.grad.detach().clone()
    per_example = []
    for x, y in zip(X, Y):
        model.zero_grad()
        criterion(model(x[None]), y[None]).backward()
        per_example.append(model.hidden.weight.grad.detach().clone())
    mean_error = float((torch.stack(per_example).mean(0) - gradient).abs().max())
    epsilon = 1e-5
    numerical = torch.empty_like(gradient)
    with torch.no_grad():
        for r in range(2):
            for c in range(2):
                original = model.hidden.weight[r, c].item()
                model.hidden.weight[r, c] = original + epsilon
                plus = float(criterion(model(X), Y))
                model.hidden.weight[r, c] = original - epsilon
                minus = float(criterion(model(X), Y))
                model.hidden.weight[r, c] = original
                numerical[r, c] = (plus - minus) / (2 * epsilon)
    finite_error = float((numerical - gradient).abs().max())
    assert mean_error < 1e-12 and finite_error < 1e-7
    return {"autograd": gradient.tolist(), "finite_difference": numerical.tolist(),
            "finite_difference_max_error": finite_error, "mean_gradient_max_error": mean_error}


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {"settings": {"seed": SEED, "steps": STEPS, "learning_rate": LEARNING_RATE,
                            "optimiser": "Adam", "dtype": "float64", "device": "cpu"}}
    histories = {}
    for activation in ("sigmoid", "tanh", "relu"):
        model, result, history = train(activation)
        results[activation], histories[activation] = result, history
    assert results["tanh"]["all_correct"] and results["tanh"]["final_loss"] < 0.01
    torch.manual_seed(SEED)
    results["gradient_check"] = check_gradient(Network("tanh"))
    _, results["zero_weights_random_biases"], _ = train("tanh", zero_weights=True)
    # an additional experiment ties biases as well, isolating exact unit symmetry.
    torch.manual_seed(SEED)
    symmetric = Network("tanh")
    with torch.no_grad():
        for parameter in symmetric.parameters():
            parameter.zero_()
    optimiser = torch.optim.Adam(symmetric.parameters(), lr=LEARNING_RATE)
    tied_snapshots = []
    for step in range(100):
        optimiser.zero_grad()
        loss = nn.BCEWithLogitsLoss()(symmetric(X), Y)
        loss.backward()
        optimiser.step()
        assert torch.equal(symmetric.hidden.weight[0], symmetric.hidden.weight[1])
        if step in (0, 1, 10, 99):
            tied_snapshots.append({"step": step + 1,
                                  "weights": symmetric.hidden.weight.detach().tolist()})
    results["zero_weights_and_biases"] = {"snapshots": tied_snapshots,
                                         "final_loss": float(loss.detach()), "identical_rows": True}
    # compare a single affine classifier to the nonlinear baseline.
    torch.manual_seed(SEED)
    linear = nn.Linear(2, 1, dtype=torch.float64)
    optimiser = torch.optim.Adam(linear.parameters(), lr=LEARNING_RATE)
    for _ in range(STEPS):
        optimiser.zero_grad()
        loss = nn.BCEWithLogitsLoss()(linear(X), Y)
        loss.backward()
        optimiser.step()
    results["linear"] = {"final_loss": float(nn.BCEWithLogitsLoss()(linear(X), Y).detach()),
                         "probabilities": torch.sigmoid(linear(X)).detach().flatten().tolist()}
    multi, results["three_class"], _ = train("tanh", outputs=3)
    assert results["three_class"]["all_correct"]
    with torch.no_grad():
        logits = multi(X)
        p = torch.softmax(logits, dim=1)
        shift_error = float((p - torch.softmax(logits + 100, dim=1)).abs().max())
        sum_error = float((p.sum(1) - 1).abs().max())
    assert shift_error < 1e-12 and sum_error < 1e-12
    results["three_class"].update(softmax_sum_error=sum_error, softmax_shift_error=shift_error,
                                   output_weight_shape=list(multi.output.weight.shape))
    results["repeat_seeds"] = {str(seed): train("tanh", seed=seed)[1]
                               for seed in (0, 1, 2)}
    folder = Path(__file__).resolve().parent
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for activation, values in histories.items():
        axes[0].plot(values, label=activation, linewidth=1)
    axes[0].set(xlabel="Training step", ylabel="Mean binary cross-entropy", yscale="log")
    axes[0].legend(fontsize=8)
    points = X.numpy()
    axes[1].scatter(points[:, 0], points[:, 1], c=Y.numpy().flatten(), cmap="coolwarm", s=65)
    for point, label in zip(points, Y.numpy().flatten()):
        axes[1].annotate(f"y={int(label)}", point, xytext=(4, 4), textcoords="offset points")
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", xticks=[0, 1], yticks=[0, 1],
                xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
    fig.tight_layout()
    fig.savefig(folder / "experiments.png", dpi=180)
    plt.close(fig)
    for key in ("sigmoid", "tanh", "relu", "three_class"):
        result = results[key]
        print(key, "loss", result["final_loss"], "labels", result["predictions"],
              "correct", result["all_correct"])
    print("Gradient, averaging, symmetry, and softmax checks passed.")


if __name__ == "__main__":
    main()
