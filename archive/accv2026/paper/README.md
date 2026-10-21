# TriPoseFusion ACCV Paper

This directory contains the ACCV manuscript source for the TriPoseFusion paper.

## Main Files

- `main.tex`: manuscript source.
- `egbib.bib`: bibliography entries.
- `figures/`: PDF figures included by `main.tex`.
- `llncs.cls`, `ruler.sty`, `splncs.bst`, `cvpr_eso.sty`, `eso-pic.sty`: conference/template support files.

## Supporting Directories

- `figures/source/`: editable EPS source versions of generated figures.
- `archive/`: old manuscript/PDF snapshots kept for reference; not used by the current manuscript.
- `template_assets/`: original ACCV template example assets; not used by the current manuscript.

## Build Note

The current manuscript references PDF figures, so a standard `pdflatex` workflow should work once a TeX distribution is available.
