PDFS=stellarator_workflow.pdf stellarator_io_reference.pdf
BIB=references.bib

.PHONY: all clean

all: $(PDFS)

%.pdf: %.tex $(BIB)
	pdflatex -interaction=nonstopmode $<
	bibtex $(basename $<)
	pdflatex -interaction=nonstopmode $<
	pdflatex -interaction=nonstopmode $<

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc
