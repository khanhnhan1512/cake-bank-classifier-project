# Cake Bank Classifier Project

A small image classification project and codebase for preprocessing and training a classifier on a custom dataset (notebooks, data loaders, transforms, and evaluation utilities included).

## Dataset characteristics
- Images may be flipped (vertical/horizontal).
- Mixed aspect ratios and sizes (square and rectangular images).
- Subjects in the images wear masks.
- Images captured from multiple viewpoints (angled, straight-on, etc.).

## Repository structure

Top-level files
- `pyproject.toml`, `requirements.txt` — project metadata and Python dependencies.
- `task_2.md` — task description / notes.

Key directories
- `notebooks/` — exploratory and baseline notebooks (`baseline.ipynb`, `baseline_colab.ipynb`).
- `src/` — source code:
	- `preprocess.py` — dataset preprocessing entrypoint.
	- `classifier/` — classifier package:
		- `data_module/` — dataset, dataloader and transform implementations.
		- `evaluations.py` — evaluation utilities.
		- `loss_functions.py` — custom loss implementations.
		- `regularizations.py` — regularization helpers.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run preprocessing (if applicable):

```bash
python src/preprocess.py
```

3. Open and run the notebooks for baseline experiments:

```bash
jupyter lab notebooks/
```

4. Use the code in `src/classifier` to build training and evaluation scripts. The package provides:
- Dataset and DataLoader implementations in `src/classifier/data_module/` (use these to create PyTorch dataloaders).
- Utility functions for evaluation in `src/classifier/evaluations.py`.

Example (importing dataloader in a script):

```py
from classifier.data_module.dataloader import create_dataloader

train_loader = create_dataloader(...)
for batch in train_loader:
		images, labels = batch
		# train step
```

## Development notes
- Notebooks provide baseline experiments and are a good starting point for model development.
- Keep data preprocessing deterministic where possible (seed transforms) to ease debugging.

## Tests & linting
This repository doesn't include automated tests yet. Before contributing, run linters or type-checkers if you add new modules.

## Contributing
- Fork the repo, create a feature branch, and open a pull request.
- Provide a short description of changes and, when possible, a minimal test or notebook showing the effect.

## License
This project doesn't include an explicit license file. Add `LICENSE` if you intend to publish or share the code publicly.

## Contact
For questions about the code or data, open an issue in this repository.

---
If you'd like, I can also:
- add a runnable training script (train.py) that uses the dataloader and a simple CNN,
- add a minimal `requirements.txt` (or pin versions) or a `Makefile` to automate common tasks,
or
- translate README content into Vietnamese — tell me which you prefer.
