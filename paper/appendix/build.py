"""Build the appendix: generated tables + example board -> appendix.pdf, and one PNG per table (for the Google Doc).

    python -m coordination_game.stats          # (re)generates paper/appendix/tables/stat_*.tex from the runs
    python paper/appendix/build.py             # board table, compile appendix.pdf, render png/table_*.png

Needs tectonic and pdftoppm (conda install -c conda-forge tectonic poppler).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BIN = Path(shutil.which("tectonic") or Path.home() / "miniforge3/envs/coordgame/bin/tectonic").parent


def num_tex(x: str) -> str:
    """Board item -> LaTeX math, e.g. '√5' -> $\\sqrt{5}$, '9.34 × 10^7' -> $9.34\\times 10^{7}$."""
    s = x
    s = re.sub(r" × 10\^(-?\d+)", r"\\times 10^{\1}", s)
    s = re.sub(r"∛(\d+)", r"\\sqrt[3]{\1}", s)
    s = re.sub(r"√(\d+)", r"\\sqrt{\1}", s)
    s = s.replace("π", r"\pi ").replace("ln ", r"\ln ").replace("log₂", r"\log_2 ").replace("e²", "e^{2}")
    return f"${s.strip()}$"


def board_table():
    boards = json.loads((ROOT / "data/boards_v2.json").read_text())
    b = boards[0]["items"]
    rows = [b[i:i + 5] for i in range(0, 20, 5)]
    body = [r"\begin{tabular}{ccccc}", r"\toprule"]
    body += [" & ".join(num_tex(x) for x in r) + r" \\" for r in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables/board0.tex").write_text("\n".join(body) + "\n")


def tex(cmd, cwd):
    subprocess.run([str(BIN / "tectonic"), "-X", "compile", cmd], cwd=cwd, check=True)


def table_pngs():
    """Each generated table as its own cropped PNG (300 dpi), for pasting into the Google Doc."""
    out = HERE / "png"
    out.mkdir(exist_ok=True)
    work = HERE / "_standalone"
    work.mkdir(exist_ok=True)
    for t in sorted((HERE / "tables").glob("*.tex")):
        doc = (r"\documentclass[border=6pt]{standalone}" "\n" r"\usepackage{booktabs}\usepackage{amsmath}"
               r"\usepackage{fontspec}\setmainfont{texgyrepagella}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold]" "\n"
               r"\begin{document}" "\n" + t.read_text() + r"\end{document}" "\n")
        (work / f"{t.stem}.tex").write_text(doc)
        tex(f"{t.stem}.tex", work)
        subprocess.run([str(BIN / "pdftoppm"), "-png", "-r", "300", "-singlefile", f"{t.stem}.pdf", str(out / t.stem)],
                       cwd=work, check=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    board_table()
    tex("appendix.tex", HERE)
    table_pngs()
    print("built", HERE / "appendix.pdf", "and", HERE / "png")
