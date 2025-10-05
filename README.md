# Urban Perception Benchmark

Code, prompts, and an evaluation harness for the paper:

> **“Do Vision–Language Models See Urban Scenes as People Do? An Urban Perception Benchmark.”**

Dataset (images + annotations) is hosted on Hugging Face:

[Urban Perception Benchmark](https://huggingface.co/datasets/rsdmu/urban-perception-benchmark)

---

## Quickstart

Requires Python and `pip`.

### Setup

```bash
# Create and activate a virtual environment (Unix/macOS)
python -m venv .venv
source .venv/bin/activate

# (Windows PowerShell)
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Download the Dataset

```python
git lfs install
git clone https://huggingface.co/datasets/rsdmu/urban-perception-benchmark
```

---


## Citation

If you find this repository or dataset useful, please cite:

```bibtex
@misc{mushkani2025visionlanguagemodelsurbanscenes,
  title        = {Do Vision--Language Models See Urban Scenes as People Do? An Urban Perception Benchmark},
  author       = {Rashid Mushkani},
  year         = {2025},
  url          = {https://arxiv.org/abs/2509.14574}
}
```

