PDFLATEXOPTS = -file-line-error -interaction=nonstopmode -halt-on-error -synctex=1

all: slides.pdf

.venv/bin/python:
	python3 -m venv .venv
	./.venv/bin/python -m pip install -r requirements.txt

slides.pdf: slides.tex .venv/bin/python $(wildcard images/*.py images/*.tex images/*.pdf images/*.tikz images/*.png) arlwide_theme/theme.tex
	$(MAKE) -C images all
	pdflatex $(PDFLATEXOPTS) slides
	pdflatex $(PDFLATEXOPTS) slides


define HANDOUT_PYSCRIPT
import sys
from PyPDF2 import PdfReader, PdfWriter
pdf_reader = PdfReader(sys.argv[-2])
pdf_writer = PdfWriter()
pages = [1, 4, 6, 10, 15, 20, 22, 24, 34, 35, 36, 37, 38, 39, 41, 47, 49, 51]
for idx in pages:
    pdf_writer.add_page(pdf_reader.pages[idx-1])
pdf_writer.write(sys.argv[-1])
pdf_writer.close()
endef
export HANDOUT_PYSCRIPT


handout.pdf: slides.pdf
	.venv/bin/python -c "$$HANDOUT_PYSCRIPT" $< $@


png: png/slides_01.png

png/slides_01.png: slides.pdf
	@mkdir -p png
	convert -density 600 $< png/slides_%02d.png

pdflatex:
	@echo "Compiling Main File ..."
	pdflatex $(PDFLATEXOPTS) slides
	@echo "Done"

update:
	pdflatex $(PDFLATEXOPTS) slides

clean:
	@echo "Cleaning up files from LaTeX compilation ..."
	$(MAKE) -C images clean
	rm -f *.aux
	rm -f *.log
	rm -f *.toc
	rm -f *.bbl
	rm -f *.blg
	rm -rf *.out
	rm -f *.bak
	rm -f *.ilg
	rm -f *.snm
	rm -f *.nav
	rm -f *.fls
	rm -f *.table
	rm -f *.gnuplot
	rm -f *.fdb_latexmk
	rm -f *.synctex.gz
	@echo "Done"

distclean: clean
	$(MAKE) -C images distclean
	rm -rf .venv
	rm -rf png
	rm -f slides.pdf
	rm -f handout.pdf

.PHONY: all pdflatex pdf png clean distclean
