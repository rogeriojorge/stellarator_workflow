PDF=stellarator_workflow.pdf
TEX=stellarator_workflow.tex
BIB=references.bib

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex stellarator_workflow
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc
