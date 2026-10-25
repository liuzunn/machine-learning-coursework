# -*- coding: utf-8 -*-
"""把 nb_part1~4 的 CELLS 组装成最终 notebook"""
import importlib.util
import os
import sys
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "B_XXXXXXXXXX_XXX_作业2_代码.ipynb")

parts = ["nb_part1.py", "nb_part2.py", "nb_part3.py", "nb_part4.py"]
all_cells = []
for p in parts:
    path = os.path.join(HERE, p)
    spec = importlib.util.spec_from_file_location(p[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"{p}: {len(mod.CELLS)} 个单元格")
    all_cells.extend(mod.CELLS)

nb = new_notebook()
nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src)
            for kind, src in all_cells]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.9",
                      "mimetype": "text/x-python", "file_extension": ".py",
                      "pygments_lexer": "ipython3", "nbconvert_exporter": "python"},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("写出:", OUT)
print("总单元格数:", len(nb.cells), "（markdown", sum(1 for c in nb.cells if c.cell_type == "markdown"),
      "/ code", sum(1 for c in nb.cells if c.cell_type == "code"), "）")
