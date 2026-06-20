#!/usr/bin/env bash
# Build the arXiv source tarball for the chaos manuscript.
#
# arXiv compiles the .tex itself, so the tarball ships only source: the main
# .tex (its bibliography is an inline `thebibliography`, so there is no .bbl/.bib
# to include) plus the 10 figure PDFs. The manuscript's \graphicspath includes
# both `figures/` (this package's layout) and `figures/chaos/` (the main repo),
# so it builds in either place.
#
# Usage:  ./build_arxiv.sh        -> writes chaos_arxiv_source.tar.gz
set -euo pipefail
cd "$(dirname "$0")"

STAGE=arxiv_src
rm -rf "$STAGE"
mkdir -p "$STAGE/figures"
cp chaos_universality_classes.tex "$STAGE/"
cp figures/*.pdf "$STAGE/figures/"

# sanity: the staged bundle must compile on its own
( cd "$STAGE" && pdflatex -interaction=nonstopmode chaos_universality_classes.tex >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode chaos_universality_classes.tex >/dev/null 2>&1 )
echo "staged bundle compiles ($(cd "$STAGE" && grep -c '' chaos_universality_classes.tex) lines of tex, $(ls "$STAGE"/figures | wc -l | tr -d ' ') figures)"

# strip build artifacts, tar the source only
rm -f "$STAGE"/*.aux "$STAGE"/*.log "$STAGE"/*.out "$STAGE"/*.pdf
tar czf chaos_arxiv_source.tar.gz -C "$STAGE" .
echo "wrote chaos_arxiv_source.tar.gz ($(du -h chaos_arxiv_source.tar.gz | cut -f1))"
