"""Builds notebooks/Final_Experiments_Bread_Dairy.ipynb from src/final_experiments.py (# %% cells)."""
import json, re, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "src" / "final_experiments.py").read_text(encoding="utf-8")
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    if header.strip().startswith("[markdown]"):
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.strip("\n")})
cells.insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": "!pip -q install lightgbm statsmodels pytorch-forecasting lightning"})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
out = root / "notebooks" / "Final_Experiments_Bread_Dairy.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(cells), "cells")
