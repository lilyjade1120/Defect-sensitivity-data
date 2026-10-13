#!/usr/bin/gnuplot -persist
set xtics ( "M" 0,  "Y" 50,  "Gamma" 100,  "X" 150,  "M" 200,  "Gamma" 271 )
unset key
nRows = real(system("awk '$1==\"kpoint\" {nRows++} END {print nRows}' bandstruct.kpoints"))
nCols = real(system("wc -c < bandstruct.eigenvals")) / (8*nRows)
formatString = system(sprintf("echo '' | awk 'END { str=\"\"; for(i=0; i<%d; i++) str = str \"%%\" \"lf\"; print str}'", nCols))
plot for [i=1:nCols] "bandstruct.eigenvals" binary format=formatString u 0:i w l
