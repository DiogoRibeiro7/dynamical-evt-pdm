.PHONY: paper supplement clean

# The main document resolves supplementary table numbers through xr, which reads
# supplement/supplement.aux, so the supplement is built first.
paper: supplement
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

supplement:
	cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

clean:
	latexmk -C main.tex
	cd supplement && latexmk -C supplement.tex
