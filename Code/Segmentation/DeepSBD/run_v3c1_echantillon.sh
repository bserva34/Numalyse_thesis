#!/bin/bash

for i in {4..5}
do
    echo "Traitement de l'échantillon $i"

    python3 deep_sbd.py ../../../Dataset/Dataset_Shot/V3C/V3C1/echantillons/echantillon_${i}.txt

    echo "Exécution de copie_doublon.py"
    python3 ../Experimentation/copie_doublon.py

    echo "----------------------------------------"
done
