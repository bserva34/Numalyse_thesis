#!/bin/bash


#METHOD_PATH=("AutoShot_TEST/Adaptive" "AutoShot_TEST/AutoShot" "AutoShot_TEST/Bi" "AutoShot_TEST/Bi_fenetre" "AutoShot_TEST/DeepSBD" "AutoShot_TEST/Suguna")

#METHOD_PATH=("ClipShots_TEST/Adaptive" "ClipShots_TEST/AutoShot" "ClipShots_TEST/Suguna")

#METHOD_PATH=("Adaptive_V3C1_bis" "AutoShot_V3C1" "Bi_V3C1" "DeepSBD_V3C1_bis" "Suguna_V3C1")

METHOD_PATH=("BBC_Fade/TransnetV2_cinema"
"BBC_FadeBlack/TransnetV2_cinema"
"BBC_FadeBlack_mirror/TransnetV2_cinema"
"BBC_TEST/TransnetV2_cinema")

METHOD_PATH=("BBC_TEST/Adaptive" 
"BBC_TEST/AutoShot" 
"BBC_TEST/Bi" 
"BBC_TEST/Bi_fenetre" 
"BBC_TEST/DeepSBD" 
"BBC_TEST/Suguna"
"BBC_TEST/TransnetV2"
"BBC_Fade/Adaptive" 
"BBC_Fade/AutoShot" 
"BBC_Fade/Bi" 
"BBC_Fade/Bi_fenetre" 
"BBC_Fade/DeepSBD" 
"BBC_Fade/Suguna"
"BBC_Fade/TransnetV2"
"BBC_FadeBlack/Adaptive" 
"BBC_FadeBlack/AutoShot" 
"BBC_FadeBlack/Bi" 
"BBC_FadeBlack/Bi_fenetre" 
"BBC_FadeBlack/DeepSBD" 
"BBC_FadeBlack/Suguna"
"BBC_FadeBlack/TransnetV2"
"BBC_FadeBlack_mirror/Adaptive" 
"BBC_FadeBlack_mirror/AutoShot" 
"BBC_FadeBlack_mirror/Bi" 
"BBC_FadeBlack_mirror/Bi_fenetre" 
"BBC_FadeBlack_mirror/DeepSBD" 
"BBC_FadeBlack_mirror/Suguna"
"BBC_FadeBlack_mirror/TransnetV2"
"BBC_Global/Adaptive" 
"BBC_Global/AutoShot" 
"BBC_Global/Bi" 
"BBC_Global/Bi_fenetre" 
"BBC_Global/DeepSBD" 
"BBC_Global/Suguna"
"BBC_Global/TransnetV2"
"BBC_Dissolve/Adaptive" 
"BBC_Dissolve/AutoShot" 
"BBC_Dissolve/Bi" 
"BBC_Dissolve/Bi_fenetre" 
"BBC_Dissolve/DeepSBD" 
"BBC_Dissolve/Suguna"
"BBC_Dissolve/TransnetV2"
)


METHOD_PATH=(
"BBC_Dissolve/Adaptive" 
"BBC_Dissolve/Suguna"
"BBC_Dissolve/Bi" 
"BBC_Dissolve/Bi_fenetre" 
"BBC_Dissolve/DeepSBD" 
"BBC_Dissolve/TransnetV2"
"BBC_Dissolve/AutoShot" 
)



#METHOD_PATH=("BBC_TEST/Adaptive" "BBC_TEST/AutoShot" "BBC_TEST/Bi" "BBC_TEST/Bi_fenetre" "BBC_TEST/DeepSBD" "BBC_TEST/Suguna")
#METHOD_PATH=("BBC_Fade/Adaptive" "BBC_Fade/AutoShot" "BBC_Fade/Bi" "BBC_Fade/Bi_fenetre" "BBC_Fade/DeepSBD" "BBC_Fade/Suguna")
#METHOD_PATH=("BBC_FadeBlack/Adaptive" "BBC_FadeBlack/AutoShot" "BBC_FadeBlack/Bi" "BBC_FadeBlack/Bi_fenetre" "BBC_FadeBlack/DeepSBD" "BBC_FadeBlack/Suguna")
#METHOD_PATH=("BBC_FadeBlack_mirror/Adaptive" "BBC_FadeBlack_mirror/AutoShot" "BBC_FadeBlack_mirror/Bi" "BBC_FadeBlack_mirror/Bi_fenetre" "BBC_FadeBlack_mirror/DeepSBD" "BBC_FadeBlack_mirror/Suguna")
#METHOD_PATH=("BBC_Global/Adaptive" "BBC_Global/AutoShot" "BBC_Global/Bi" "BBC_Global/Bi_fenetre" "BBC_Global/DeepSBD" "BBC_Global/Suguna")




for t in {6..6}
do
	for i in "${METHOD_PATH[@]}"
	do
		echo "Calcul Perf pour la méthode $i avec une tolérance de $t"

	    python3 calcul_perf.py "$i" "$t"
	done
done
