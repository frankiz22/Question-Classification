Proposez un algorithme permettant, étant donné un verbatim en entrée, de classifier l’intention exprimée. Nous nous placerons dans le cas d’un chatbot d’assistance sur le site d’une grande agence de tourisme. Les différentes classes sont les suivantes :
translate: l’utilisateur souhaite traduire une phrase dans une autre langue
travel_alert: l’utilisateur demande si sa destination est concernée par une alerte de voyage
flight_status: l’utilisateur demande des informations sur le statut de son vol
lost_luggage: l’utilisateur signale la perte de ses bagages
travel_suggestion: l’utilisateur souhaite une recommandation de voyage
carry_on: l'utilisateur souhaite des informations sur les bagages à main
book_hotel: l’utilisateur souhaite réserver un hôtel
book_flight: l’utilisateur souhaite réserver un vol

L’agence vous signale que l’intention lost_luggage redirige vers un service client avec des conseillers téléphoniques ayant un coût élevé.
De plus, il est possible que l’utilisateur formule une demande dite “hors-scope” c’est-à-dire qui n’appartient à aucune des 8 classes ci-dessus. Dans ce cas, on attend que l’algorithme renvoie une réponse spéciale out_of_scope.
