# CS F407 AI Labs

Done: agents, search, logical planning, n-gram models, and neural models.
The local LLM/BN inference, estimation, and validation are also done.

## Run

Use Python 3.11. From this directory, create and activate a shared `.venv`, install dependencies, and run a lab:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python agents_lab/solution.py
```

Replace `agents_lab` with `search_lab`, `logic_lab`, `bn_lab`,
`neur_models_lab`, or `llm_bn_lab` to run another lab.

## TODO

- `logic_lab`: optional Prolog checks.
- `llm_bn_lab`: Qwen code generation and comparison.
- `transformers_lab`: pretrained model demonstrations and Ollama.
