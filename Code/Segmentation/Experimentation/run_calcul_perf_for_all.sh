#!/bin/bash

#METHOD_PATH=("BBC_TEST/Adaptive_BBC" "BBC_TEST/AutoShot_BBC" "BBC_TEST/Bi_BBC" "BBC_TEST/DeepSBD_BBC" "BBC_TEST/Suguna_BBC")

METHOD_PATH=("AutoShot_TEST/Adaptive_AutoShot" "AutoShot_TEST/AutoShot_AutoShot" "AutoShot_TEST/Bi_AutoShot" "AutoShot_TEST/DeepSBD_AutoShot" "AutoShot_TEST/Suguna_AutoShot")

#METHOD_PATH=("Adaptive_V3C1_bis" "AutoShot_V3C1" "Bi_V3C1" "DeepSBD_V3C1_bis" "Suguna_V3C1")

for t in {0..12}
do
	for i in "${METHOD_PATH[@]}"
	do
		echo "Calcul Perf pour la méthode $i avec une tolérance de $t"

	    python3 calcul_perf.py "$i" "$t"
	done
done
