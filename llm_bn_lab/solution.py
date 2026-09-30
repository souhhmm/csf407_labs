import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/csf407-matplotlib")
from itertools import product
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.parameter_estimator import DiscreteMLE, DiscreteBayesianEstimator

EDGES = [("Cloudy", "Rain"), ("Cloudy", "Sprinkler"),
         ("Rain", "WetGrass"), ("Sprinkler", "WetGrass")]
VARIABLES = ["Cloudy", "Rain", "Sprinkler", "WetGrass"]
STATES = {name: [0, 1] for name in VARIABLES}
QUERIES = [("Rain", {"WetGrass": 1}), ("Sprinkler", {"WetGrass": 1}),
           ("Cloudy", {"WetGrass": 1}), ("Rain", {"WetGrass": 1, "Sprinkler": 0})]


def build_model():
    model = DiscreteBayesianNetwork(EDGES)
    model.add_cpds(
        TabularCPD("Cloudy", 2, [[0.5], [0.5]], state_names={"Cloudy": [0, 1]}),
        TabularCPD("Rain", 2, [[0.8, 0.2], [0.2, 0.8]],
                   evidence=["Cloudy"], evidence_card=[2],
                   state_names={"Rain": [0, 1], "Cloudy": [0, 1]}),
        TabularCPD("Sprinkler", 2, [[0.5, 0.9], [0.5, 0.1]],
                   evidence=["Cloudy"], evidence_card=[2],
                   state_names={"Sprinkler": [0, 1], "Cloudy": [0, 1]}),
        TabularCPD("WetGrass", 2, [[0.99, 0.10, 0.10, 0.01], [0.01, 0.90, 0.90, 0.99]],
                   evidence=["Rain", "Sprinkler"], evidence_card=[2, 2],
                   state_names={"Rain": [0, 1], "Sprinkler": [0, 1], "WetGrass": [0, 1]}),
    )
    assert model.check_model()
    return model


def joint_probability(c, r, s, w):
    pr = (0.2, 0.8)[c]
    ps = (0.5, 0.1)[c]
    pw = {(0, 0): 0.01, (0, 1): 0.90, (1, 0): 0.90, (1, 1): 0.99}[(r, s)]
    return 0.5 * (pr if r else 1 - pr) * (ps if s else 1 - ps) * (pw if w else 1 - pw)


def enumerate_posterior(query, evidence):
    numerator = denominator = 0.0
    for values in product([0, 1], repeat=4):
        assignment = dict(zip(VARIABLES, values))
        if all(assignment[name] == state for name, state in evidence.items()):
            probability = joint_probability(*values)
            denominator += probability
            numerator += probability * assignment[query]
    return numerator / denominator


def fit(data, bayesian=False):
    estimator = DiscreteBayesianEstimator(state_names=STATES, prior_type="BDeu",
                                          equivalent_sample_size=10) if bayesian else DiscreteMLE(state_names=STATES)
    model = DiscreteBayesianNetwork(EDGES)
    model.fit(data, estimator=estimator)
    assert model.check_model()
    assert set(model.edges()) == set(EDGES)
    return model


def serialize_cpds(model):
    return {cpd.variable: {"parents": cpd.variables[1:], "values": cpd.get_values().tolist()}
            for cpd in model.get_cpds()}


def rain_estimate(model):
    return float(model.get_cpds("Rain").get_value(Rain=1, Cloudy=1))


def mle_count_check(data, model):
    max_error = 0.0
    for cpd in model.get_cpds():
        parents = cpd.variables[1:]
        for values in product([0, 1], repeat=len(parents)):
            condition = dict(zip(parents, values))
            selected = data
            for name, value in condition.items():
                selected = selected.loc[selected[name] == value]
            for state in (0, 1):
                expected = float((selected[cpd.variable] == state).mean()) if len(selected) else 0.5
                measured = float(cpd.get_value(**{cpd.variable: state, **condition}))
                max_error = max(max_error, abs(measured - expected))
    assert max_error < 1e-12
    return max_error


def bayesian_count_check(data, model):
    max_error = 0.0
    for cpd in model.get_cpds():
        parents = cpd.variables[1:]
        configurations = 2 ** len(parents)
        pseudocount = 10 / (2 * configurations)
        for values in product([0, 1], repeat=len(parents)):
            condition = dict(zip(parents, values))
            selected = data
            for name, value in condition.items():
                selected = selected.loc[selected[name] == value]
            for state in (0, 1):
                count = int((selected[cpd.variable] == state).sum())
                expected = (count + pseudocount) / (len(selected) + 2 * pseudocount)
                measured = float(cpd.get_value(**{cpd.variable: state, **condition}))
                max_error = max(max_error, abs(measured - expected))
    assert max_error < 1e-12
    return max_error


def main():
    model = build_model()
    assert set(model.nodes()) == set(VARIABLES)
    assert set(model.edges()) == set(EDGES)
    assert abs(sum(joint_probability(*values) for values in product([0, 1], repeat=4)) - 1) < 1e-12
    inference = VariableElimination(model)
    results = {"todo": ["Qwen code generation and comparison with the reference implementation"],
               "cpds": serialize_cpds(model), "queries": []}
    for query, evidence in QUERIES:
        measured = float(inference.query([query], evidence=evidence, show_progress=False).values[1])
        expected = enumerate_posterior(query, evidence)
        assert abs(measured - expected) < 1e-12
        results["queries"].append({"variable": query, "evidence": evidence,
                                   "pgmpy": measured, "enumeration": expected,
                                   "absolute_error": abs(measured - expected)})
    folder = Path(__file__).resolve().parent
    data = model.simulate(n_samples=1000, seed=7, show_progress=False)
    fitted = fit(data)
    results["mle_1000"] = {"cpds": serialize_cpds(fitted),
                           "rain_given_cloudy": rain_estimate(fitted),
                           "count_max_error": mle_count_check(data, fitted)}
    results["sample_sizes"] = []
    for n in (20, 50, 100, 500, 1000, 5000):
        sample = model.simulate(n_samples=n, seed=7, show_progress=False)
        estimate = rain_estimate(fit(sample))
        results["sample_sizes"].append({"n": n, "estimate": estimate, "absolute_error": abs(estimate - 0.8)})
    small = model.simulate(n_samples=30, seed=11, show_progress=False)
    mle, bayes = fit(small), fit(small, bayesian=True)
    results["sparse_data"] = {"mle": rain_estimate(mle), "bdeu": rain_estimate(bayes),
                              "cpds_mle": serialize_cpds(mle), "cpds_bdeu": serialize_cpds(bayes),
                              "mle_count_error": mle_count_check(small, mle),
                              "bdeu_count_error": bayesian_count_check(small, bayes)}
    results["seeds"] = []
    for seed in (1, 2, 3, 4, 5):
        sample = model.simulate(n_samples=100, seed=seed, show_progress=False)
        results["seeds"].append({"seed": seed, "estimate": rain_estimate(fit(sample))})
    broken = DiscreteBayesianNetwork([("Cloudy", "Rain")])
    broken.add_cpds(TabularCPD("Cloudy", 2, [[0.5], [0.5]]),
                    TabularCPD("Rain", 2, [[0.9, 0.2], [0.3, 0.8]],
                               evidence=["Cloudy"], evidence_card=[2]))
    try:
        broken.check_model()
    except ValueError as error:
        results["invalid_cpd"] = str(error)
    else:
        raise AssertionError("Non-normalised CPT was accepted")
    wrong = build_model()
    wrong.remove_cpds(wrong.get_cpds("WetGrass"))
    wrong.add_cpds(TabularCPD("WetGrass", 2,
                             [[0.99, 0.1, 0.01, 0.1], [0.01, 0.9, 0.99, 0.9]],
                             evidence=["Rain", "Sprinkler"], evidence_card=[2, 2]))
    assert wrong.check_model()
    wrong_result = float(VariableElimination(wrong).query(["Rain"], evidence={"WetGrass": 1},
                                                        show_progress=False).values[1])
    assert abs(wrong_result - results["queries"][0]["pgmpy"]) > 1e-3
    results["wrong_parent_order"] = {"normalised": True, "posterior": wrong_result,
                                    "trusted_posterior": results["queries"][0]["pgmpy"]}
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results["queries"]:
        print(row)
    print("N=1000 MLE:", results["mle_1000"]["rain_given_cloudy"])
    print("Sparse data:", results["sparse_data"]["mle"], results["sparse_data"]["bdeu"])
    print("All structure, normalisation, inference, estimation, and semantic checks passed.")


if __name__ == "__main__":
    main()
