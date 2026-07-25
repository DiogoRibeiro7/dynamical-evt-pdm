.PHONY: paper supplement clean

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

supplement:
	cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

clean:
	latexmk -C main.tex
	cd supplement && latexmk -C supplement.tex
