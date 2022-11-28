LATEX = latexmk
OUTPUT_DIR = _outputs
MAIN = rapport

.PHONY: all

all: pdf

pdf:
	$(LATEX) --pdf --halt-on-error $(MAIN).tex --shell-escape

watch:
	$(LATEX) --pdf --halt-on-error $(MAIN).tex --shell-escape -pvc 

clean:
	$(LATEX) -C $(MAIN).tex
