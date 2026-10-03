# fontspec requires XeLaTeX/LuaLaTeX; keep Workshop + CLI consistent.
$pdf_mode = 5;  # 5 = xelatex
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
