"""Convert a '# %%' script into a Jupyter notebook."""
import json, sys, re
src = open(sys.argv[1]).read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if "[markdown]" in header:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").split("\n"))
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))
    else:
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                          source=body.strip("\n").splitlines(True)))
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata={"accelerator": "GPU", "colab": {"provenance": []},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
json.dump(nb, open(sys.argv[2], "w"), indent=1)
print(f"{len(cells)} cells -> {sys.argv[2]}")
