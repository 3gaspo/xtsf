$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pdflatex = "C:\Users\Gaspard\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
if (-not (Test-Path -LiteralPath $pdflatex)) {
    $pdflatex = "pdflatex"
}
Push-Location $here
try {
    & $pdflatex --disable-installer -interaction=nonstopmode -halt-on-error package_summary.tex
    & $pdflatex --disable-installer -interaction=nonstopmode -halt-on-error package_summary.tex
}
finally {
    Pop-Location
}
