#!/usr/bin/env bash
# Build both PDF profiles of the paper from the same source:
#   paper_235_ieee_candidate.pdf     -- no link annotations, for IEEE PDF eXpress
#   paper_235_review_interactive.pdf -- blue DOI/ORCID/e-mail links, for review
set -euo pipefail
cd "$(dirname "$0")"
export TEXINPUTS="./cls:${TEXINPUTS:-}" BSTINPUTS="./cls:${BSTINPUTS:-}"
OUT=build/profiles; mkdir -p "$OUT"
build() {  # $1 profile, $2 output name
  local pre="\\def\\PAPERCLASSOPT{conference}\\def\\PDFPROFILE{$1}\\input{main.tex}"
  run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" -jobname="$2" "$pre" >/dev/null; }
  run "$@"; (cd "$OUT" && BSTINPUTS="../../cls:" BIBINPUTS="../..:" bibtex "$2" >/dev/null); run "$@"; run "$@"
  cp "$OUT/$2.pdf" "./$2.pdf"
  echo "$2.pdf: $(pdfinfo "$2.pdf" | awk '/^Pages/{print $2}') pages"
}
build ieee paper_235_ieee_candidate
build interactive paper_235_review_interactive
