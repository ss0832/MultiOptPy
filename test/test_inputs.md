(ver. 2025/11/06)
```
python optmain.py 222cycle.xyz -xtb GFN2-xTB -ma 70 1 9 70 11 5 70 7 3 -opt RFO_FSB -modelhess
```
```
python optmain.py claisen_rearrengment.xyz -xtb GFN2-xTB -ma 200 6 12
```
```
python optmain.py diels_alder_rxn.xyz -xtb GFN2-xTB -ma 120 1 11 120 13 8
```
```
python optmain.py epoxidation.xyz -xtb GFN2-xTB -ma 200 1 10,13 100 5 2 -opt RFO_FSB -modelhess -lcc
```
```
python optmain.py hydroboration.xyz -xtb GFN2-xTB -ma 100 1 13 100 14 4 -opt RFO_FSB -modelhess -lcc
```
```
python optmain.py  hydroformylation.xyz -xtb GFN1-xTB -ma 150 2 9  -opt RFO_FSB -modelhess
```
```
python optmain.py  intramolecular_aldol_rxn.xyz -xtb GFN2-xTB -ma 100 2 16 100 17 25 -opt RFO_FSB -modelhess -lcc
```
```
python optmain.py reductive_elimination.xyz -xtb GFN1-xTB -ma 200 14 25 -opt RFO_FSB -modelhess -lcc
```
```
python optmain.py swarn_oxidation.xyz -xtb GFN2-xTB -ma 50 10 6 -opt RFO_FSB -modelhess
```
```
python optmain.py witting_rxn.xyz -xtb GFN2-xTB -ma 100 1 26 150 14 24
```
```
python ieipmain.py curtius_rearrgement -xtb GFN2-xTB
```
```
python run_autots.py autots_v2_test.py -cfg config_autots_v2_test.json
```
```
python run_autots.py aldol_rxn.py -cfg config_autots_run_xtb_test.json
```