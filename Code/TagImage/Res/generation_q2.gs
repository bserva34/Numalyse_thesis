function creerFormulaireImages() {
  // ================= CONFIGURATION =================
  // Remplissez ici l'identifiant (ID) de votre dossier Drive contenant les vidéos
  var ID_DOSSIER_VIDEOS = "17FQvw3PDGxjmgCkxcS766cnMEFHhisPS"; 
  // =================================================

  // 1. Récupération des vidéos du Drive
  var dossier = DriveApp.getFolderById(ID_DOSSIER_VIDEOS);
  var fichiers = dossier.getFiles();
  var listeVideos = [];

  // On parcourt le dossier pour lister toutes les vidéos
  while (fichiers.hasNext()) {
    var fichier = fichiers.next();
    listeVideos.push({
      nom: fichier.getName(),
      url: fichier.getUrl() // Récupère le lien de consultation sur Drive
    });
  }

  // Optionnel : Trier les vidéos par nom (ex: video_1, video_2...) pour qu'elles soient dans l'ordre
  listeVideos.sort(function(a, b) {
    return a.nom.localeCompare(b.nom, undefined, {numeric: true, sensitivity: 'base'});
  });

  // Sécurité : Vérifier qu'on a bien des vidéos
  if (listeVideos.length === 0) {
    Logger.log("Erreur : Aucune vidéo trouvée dans le dossier spécifié.");
    return;
  }

  // 2. Création du formulaire avec le titre principal
  var form = FormApp.create("Questionnaire Évaluation de différentes méthodes d'extraction de tag image V2");
  form.setDescription("Rappel : Une tag image est une image utilisée pour identifier un plan dans un film.\n Dans ce questionnaire, pour chaque plan que vous visionnerez, vous devrez sélectionner l’image ou les images que vous jugez réellement pertinentes pour permettre d’identifier ou de repérer ce plan (sans l’avoir sous les yeux). Il est aussi possible de n'en sélectionner aucune.\n\n Le son et les éléments textuels ne doivent pas être pris en compte dans votre choix.");
  
  // Les 6 options de réponses demandées
  var options = ["Tag-frame 1", "Tag-frame 2", "Tag-frame 3", "Tag-frame 4", "Tag-frame 5", "Tag-frame 6"];
  
  // 3. Boucle pour générer les questions (s'adapte au nombre de vidéos trouvées, max 200)
  var nombreDeQuestions = Math.min(listeVideos.length, 200); 
  
  for (var i = 0; i < nombreDeQuestions; i++) {
    var video = listeVideos[i];
    
    // On construit le titre de la question en y intégrant le lien de la vidéo
    // Format : "Question X : Sélectionnez les images permettant d'identifier le plan ci-dessous (Lien de la vidéo : [URL])"
    var titreQuestion = "Sélectionnez les images permettant d'identifier le plan ci-dessous : \n" + video.url;
    
    // Création de la question de type "Cases à cocher"0
    var item = form.addCheckboxItem();
    item.setTitle(titreQuestion);
    
    // Ajout des choix
    item.setChoices(options.map(function(choix) {
      return item.createChoice(choix);
    }));
    
    // Configuration : NON obligatoire + Réponses ALÉATOIRES
    item.setRequired(false);
    //item.setChoiceOrderRandomized(true);
  }
  
  // Affichage du lien dans les journaux
  Logger.log('Formulaire créé avec succès avec ' + nombreDeQuestions + ' questions !');
  Logger.log('URL de modification : ' + form.getEditUrl());
}