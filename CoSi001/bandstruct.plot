#!/usr/bin/gnuplot -persist
set xtics ( "Y" 0,  "Gamma" 50,  "X" 100,  "M" 150,  "Gamma" 221 )
unset key
nRows = real(system("awk '$1==\"kpoint\" {nRows++} END {print nRows}' bs.kpoints"))
nCols = real(system("wc -c < bs.eigenvals")) / (8*nRows)
formatString = system(sprintf("echo '' | awk 'END { str=\"\"; for(i=0; i<%d; i++) str = str \"%%\" \"lf\"; print str}'", nCols))
plot for [i=1:nCols] "bs.eigenvals" binary format=formatString u 0:i w l
