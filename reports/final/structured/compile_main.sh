#!/bin/bash
set -e

FILE="main_revised"

echo "1/4 First pdflatex pass..."
pdflatex -interaction=nonstopmode -file-line-error "${FILE}.tex"

echo "2/4 Running biber..."
biber "$FILE"

echo "3/4 Second pdflatex pass..."
pdflatex -interaction=nonstopmode -file-line-error "${FILE}.tex"

echo "4/4 Final pdflatex pass..."
pdflatex -interaction=nonstopmode -file-line-error "${FILE}.tex"

echo "Cleaning auxiliary files..."
rm -f \
  "${FILE}.aux" \
  "${FILE}.bcf" \
  "${FILE}.blg" \
  "${FILE}.fdb_latexmk" \
  "${FILE}.fls" \
  "${FILE}.log" \
  "${FILE}.out" \
  "${FILE}.run.xml" \
  "${FILE}.synctex.gz" \
  "${FILE}.toc"

echo "Done: ${FILE}.pdf"