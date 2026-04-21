# CardioScan — Plateforme d'Analyse ECG par Intelligence Artificielle

Application web full-stack de classification multi-labels d'ECG 12 dérivations, basée sur un réseau de neurones résiduel 1D entraîné sur le jeu de données PTB-XL.

---

## Auteurs

Projet réalisé par **Youssef Khaloufi** et **Ayoub Bourti**, sous la direction du **Pr. Mostafa Ezziyyani**.

---

## Table des matières

- [Aperçu](#aperçu)
- [Architecture du projet](#architecture-du-projet)
- [Modèle de deep learning](#modèle-de-deep-learning)
- [Prérequis](#prérequis)
- [Installation et démarrage rapide](#installation-et-démarrage-rapide)
- [Démarrage avec Docker](#démarrage-avec-docker)
- [Référence API](#référence-api)
- [Export de signal depuis Python](#export-de-signal-depuis-python)
- [Variables d'environnement](#variables-denvironnement)
- [Suggestions de déploiement](#suggestions-de-déploiement)
- [Licence](#licence)

---

## Aperçu

CardioScan est une application médicale basée sur l'IA permettant l'analyse automatique de signaux ECG à 12 dérivations. Le système classifie les enregistrements en **5 superclasses diagnostiques** avec des seuils de décision optimisés, et expose les résultats via une interface web moderne et une API REST sécurisée.

Fonctionnalités principales :

- **Modèle ResNet 1D** entraîné sur 21 799 enregistrements (PTB-XL)
- **Authentification JWT** avec gestion des utilisateurs (inscription / connexion)
- **Dashboard interactif** avec métriques du modèle et visualisations
- **Historique des analyses** par utilisateur (consultation et suppression)
- **Déploiement Docker** clé en main via `docker-compose`
- **Documentation API** auto-générée avec Swagger UI (`/docs`)

---

## Architecture du projet

```
ecg-classification-system/
│
├── backend/                        # API Python — FastAPI
│   ├── app/
│   │   ├── main.py                 # Point d'entrée + configuration CORS
│   │   ├── database.py             # Base de données SQLite asynchrone
│   │   ├── dependencies.py         # Dépendance d'authentification JWT
│   │   ├── models/
│   │   │   └── ecg_model.py        # Architecture ECGNet + ResidualBlock (PyTorch)
│   │   ├── services/
│   │   │   ├── model_service.py    # Moteur d'inférence (singleton)
│   │   │   └── auth_service.py     # Helpers bcrypt + JWT
│   │   ├── schemas/
│   │   │   └── schemas.py          # Modèles Pydantic (requêtes / réponses)
│   │   ├── routes/
│   │   │   ├── auth.py             # POST /api/auth/register | /login
│   │   │   ├── predict.py          # POST /api/predict/
│   │   │   ├── history.py          # GET | DELETE /api/history/
│   │   │   └── stats.py            # GET /api/stats/
│   │   └── model/
│   │       └── ecg_full_model.pth  # Placer le checkpoint ici
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── App.jsx                 # Configuration du routeur
│   │   ├── main.jsx                # Point d'entrée
│   │   ├── index.css               # Styles globaux + Tailwind
│   │   ├── utils/api.js            # Client Axios + tous les appels API
│   │   ├── hooks/useAuth.jsx       # Contexte d'authentification
│   │   ├── components/
│   │   │   ├── Layout.jsx          # Sidebar + outlet
│   │   │   └── ResultCard.jsx      # Affichage des résultats de prédiction
│   │   └── pages/
│   │       ├── HomePage.jsx        # Page d'accueil
│   │       ├── LoginPage.jsx       # Connexion
│   │       ├── RegisterPage.jsx    # Inscription
│   │       ├── PredictPage.jsx     # Formulaire d'analyse principal
│   │       ├── DashboardPage.jsx   # Graphiques + métriques du modèle
│   │       └── HistoryPage.jsx     # Tableau de l'historique
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── notebooks/                      # Notebooks Jupyter d'entraînement
├── data_sample/                    # Exemples de signaux ECG
├── api/                            # Version Streamlit (legacy)
├── test/                           # Tests
├── app.py                          # Application Streamlit standalone
├── docker-compose.yml
├── requirements.txt
└── sample.json                     # Exemple de payload JSON
```

---

## Modèle de deep learning

### Caractéristiques techniques

| Propriété             | Valeur                                              |
|-----------------------|-----------------------------------------------------|
| Jeu de données        | PTB-XL (21 799 enregistrements)                     |
| Entrée                | ECG 12 dérivations, 1 000 pas de temps @ 100 Hz     |
| Architecture          | 3x ResidualBlock 1D (12 -> 64 -> 128 -> 256 canaux) |
| Classifieur           | `Linear(256,128) -> ReLU -> Dropout -> Linear(128,5)` |
| Fonction de perte     | Focal Loss + `pos_weight` pour le déséquilibre      |
| Echantillonnage       | `WeightedRandomSampler`                             |
| Epoques               | 10                                                  |
| Macro AUC             | **0.917**                                           |

### Classes diagnostiques

| Code   | Nom complet                  | Seuil de décision |
|--------|------------------------------|:-----------------:|
| NORM   | ECG Normal                   | 0.55              |
| MI     | Infarctus du Myocarde        | 0.45              |
| CD     | Trouble de la Conduction     | 0.70              |
| HYP    | Hypertrophie                 | 0.65              |
| STTC   | Modification ST/T            | 0.65              |

### Performances détaillées

| Classe        | AUC       |
|---------------|-----------|
| NORM          | 0.938     |
| MI            | 0.927     |
| CD            | 0.897     |
| **Macro AUC** | **0.917** |
| **Macro F1**  | **~0.75** |

---

## Prérequis

- Python >= 3.9
- Node.js >= 18
- Docker et Docker Compose (pour le démarrage conteneurisé)
- Checkpoint du modèle : `ecg_full_model.pth`

> Sans checkpoint, l'API fonctionne en mode démo avec des poids aléatoires — tous les endpoints restent disponibles mais les prédictions sont aléatoires.

---

## Installation et démarrage rapide

### 1. Backend (FastAPI)

```bash
cd backend

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Copier le checkpoint du modèle entraîné
cp /chemin/vers/ecg_full_model.pth app/model/ecg_full_model.pth

# Lancer l'API (rechargement automatique activé)
uvicorn app.main:app --reload --port 8000
```

API disponible sur : `http://localhost:8000`  
Documentation interactive : `http://localhost:8000/docs`

### 2. Frontend (React + Vite)

```bash
cd frontend

# Installer les paquets
npm install

# Lancer le serveur de développement
npm run dev
```

Application disponible sur : `http://localhost:5173`

---

## Démarrage avec Docker

```bash
# Depuis la racine du projet
cp votre_modele.pth backend/app/model/ecg_full_model.pth

docker-compose up --build
```

| Service   | URL                         |
|-----------|-----------------------------|
| Frontend  | http://localhost:5173       |
| Backend   | http://localhost:8000       |
| API Docs  | http://localhost:8000/docs  |

---

## Référence API

### Authentification

```bash
# Inscription
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","email":"demo@example.com","password":"secret123"}'

# Connexion
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"secret123"}'
```

### Prédiction

```bash
# Générer un signal de test (Python)
python3 -c "
import numpy as np, json
sig = np.random.randn(1000, 12).tolist()
print(json.dumps({'signal_data': sig, 'patient_name': 'Patient Test', 'age': 55, 'sex': 'M'}))
" > test_payload.json

# Lancer une prédiction
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer VOTRE_TOKEN" \
  -d @test_payload.json
```

Exemple de réponse :

```json
{
  "id": 1,
  "predictions": [
    {"class": "NORM", "description": "ECG Normal",             "probability": 0.92, "positive": true},
    {"class": "MI",   "description": "Infarctus du Myocarde", "probability": 0.08, "positive": false},
    {"class": "CD",   "description": "Trouble de Conduction", "probability": 0.03, "positive": false},
    {"class": "HYP",  "description": "Hypertrophie",          "probability": 0.11, "positive": false},
    {"class": "STTC", "description": "Modification ST/T",     "probability": 0.07, "positive": false}
  ],
  "top_class": "NORM",
  "confidence": 0.92,
  "positive_classes": ["NORM"],
  "patient_name": "Patient Test",
  "age": 55,
  "sex": "M"
}
```

### Historique et statistiques (authentification requise)

```bash
TOKEN="votre_token_jwt"

# Récupérer l'historique
curl http://localhost:8000/api/history/ -H "Authorization: Bearer $TOKEN"

# Récupérer les statistiques
curl http://localhost:8000/api/stats/ -H "Authorization: Bearer $TOKEN"

# Supprimer un enregistrement
curl -X DELETE http://localhost:8000/api/history/1 -H "Authorization: Bearer $TOKEN"
```

---

## Export de signal depuis Python

```python
import numpy as np, json, requests

# X_test[0] a la forme (1000, 12)
signal = X_test[0]  # tableau numpy issu du notebook

# Méthode 1 : sauvegarder en JSON pour l'uploader frontend
with open("sample_signal.json", "w") as f:
    json.dump(signal.tolist(), f)

# Méthode 2 : envoyer directement via POST
r = requests.post(
    "http://localhost:8000/api/predict/",
    json={
        "signal_data": signal.tolist(),
        "patient_name": "Patient A",
        "age": 62,
        "sex": "F"
    },
    headers={"Authorization": "Bearer VOTRE_TOKEN"}
)
print(r.json())
```

---

## Variables d'environnement

### Backend

| Variable     | Valeur par défaut              | Description                |
|--------------|--------------------------------|----------------------------|
| `SECRET_KEY` | `change-me-in-production-...`  | Clé de signature JWT       |
| `DB_PATH`    | `cardioscan.db`                | Chemin vers la base SQLite |
| `MODEL_PATH` | `app/model/ecg_full_model.pth` | Chemin vers le checkpoint  |

### Frontend

| Variable       | Valeur par défaut       | Description              |
|----------------|-------------------------|--------------------------|
| `VITE_API_URL` | `http://localhost:8000` | URL de base de l'API     |

---

## Suggestions de déploiement

| Service     | Notes                                                                  |
|-------------|------------------------------------------------------------------------|
| **Render**  | Déployer le backend comme Web Service Docker. Tier gratuit disponible. |
| **Railway** | `railway up` depuis le dossier backend. Détecte le Dockerfile auto.   |
| **Vercel**  | Déployer le dossier `frontend/`. Configurer `VITE_API_URL`.           |
| **Fly.io**  | Idéal pour le backend PyTorch — disque persistant pour DB + modèle.   |

---

## Licence

Ce projet est distribué sous licence libre. Consultez le fichier [LICENSE.txt](./LICENSE.txt) pour plus de détails.

---

> **Avertissement :** Cet outil est destiné à des fins de recherche et d'aide à la décision. Il ne remplace pas le diagnostic d'un professionnel de santé qualifié.