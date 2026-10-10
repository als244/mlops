# GLM operation checks

```bash
python -m pip install -e '.[glm]'
python -m pytest -ra tests/glm
```

These checks use only installed MLOps APIs and the pinned, licensed reference
under `references/`. No experiment checkout, network download or model
checkpoint is needed. See [GLM APIs and scope](../../docs/GLM.md).

| Tests | Coverage |
|---|---|
| `test_api.py` | Lazy public imports and reference hash |
| `test_core.py`, `test_precision.py` | Clipped activation, routing, mHC, gradients, FP32 rules |
| `test_kda.py`, `test_block.py` | KDA recurrence, packed sequences, full/LoRA block derivatives |
| `test_indexing.py` | Causality, complete/incomplete pools and tied scores |
| `test_attention.py`, `test_mla.py`, `test_lora.py` | Sparse attention, projection absorption and external LoRA |
| `test_contracts.py` | Fake/autograd/AOT contracts and metadata reuse |

CUDA comparisons require SM80+; KDA and sparse MLA additionally require FLA
and TileLang respectively. The tested device is RTX 5090. CPU import and source
integrity checks run without a GPU. The ordinary repository suite includes
this directory and reports capability skips explicitly.
