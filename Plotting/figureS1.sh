#!/bin/bash

python3 supercellConvergence.py ../CoSi001 "CoSi(001)" \
				../CoSi111 "CoSi(111)" \
				../PdGa001 "PdGa(001)" \
				../PtAl001 "PtAl(001)" \
				../NbAs001_noSOC "NbAs(001) no SOC" \
				../NbAs001_SOC "NbAs(001) with SOC"
mv supercellConvergence.pdf figureS1.pdf
