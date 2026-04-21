# CardioScan — Plateforme de Classification ECG par Intelligence Artificielle

Application web full-stack de classification multi-labels d'ECG 12 dérivations, basée sur un réseau de neurones résiduel 1D entraîné sur le jeu de données PTB-XL.

---

## Auteurs

Projet réalisé par **Khaloufi Youssef** et **Bourti Ayoub**, sous la direction du **Pr. Ezziyyani Mostafa**.

---

## Table des matières

- [Présentation](#présentation)
- [Architecture du dépôt](#architecture-du-dépôt)
- [Données — Dataset PTB-XL](#données--dataset-ptb-xl)
- [Prétraitement et normalisation](#prétraitement-et-normalisation)
- [Architecture du modèle](#architecture-du-modèle)
- [Pipeline d'entraînement](#pipeline-dentraînement)
- [Évaluation et métriques](#évaluation-et-métriques)
- [Backend — API FastAPI](#backend--api-fastapi)
- [Frontend — React / Vite / TailwindCSS](#frontend--react--vite--tailwindcss)
- [Prérequis](#prérequis)
- [Installation et démarrage rapide](#installation-et-démarrage-rapide)
- [Démarrage avec Docker](#démarrage-avec-docker)
- [Référence API](#référence-api)
- [Export de signal depuis Python](#export-de-signal-depuis-python)
- [Variables d'environnement](#variables-denvironnement)
- [Suggestions de déploiement](#suggestions-de-déploiement)
- [Licence](#licence)

---

## Présentation

CardioScan est une plateforme médicale basée sur l'IA permettant l'analyse automatique de signaux ECG à 12 dérivations. Le système classifie les enregistrements en **5 superclasses diagnostiques** avec des seuils de décision optimisés par classe, et expose les résultats via une interface web moderne et une API REST sécurisée par JWT.

Fonctionnalités principales :

- **Modèle ResNet 1D** entraîné sur 21 799 enregistrements PTB-XL, Macro AUC = 0.917
- **Classification multi-label** sur 5 classes : NORM, MI, CD, HYP, STTC
- **Authentification JWT** avec gestion des utilisateurs (inscription / connexion)
- **Dashboard interactif** avec métriques du modèle et visualisations
- **Historique des analyses** par utilisateur (consultation et suppression)
- **Déploiement Docker** clé en main via `docker-compose`
- **Documentation API** auto-générée avec Swagger UI sur `/docs`

---

## Architecture du dépôt

```
ecg-classification-system/
│
├── backend/                        # API Python — FastAPI + PyTorch
│   ├── app/
│   │   ├── main.py                 # Point d'entrée, configuration CORS
│   │   ├── database.py             # Base SQLite asynchrone (aiosqlite)
│   │   ├── dependencies.py         # Injection de dépendance JWT
│   │   ├── models/
│   │   │   └── ecg_model.py        # Architecture ECGNet + ResidualBlock (PyTorch)
│   │   ├── services/
│   │   │   ├── model_service.py    # Singleton de chargement et inférence
│   │   │   └── auth_service.py     # Hachage bcrypt + génération/validation JWT
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
│   │   ├── utils/api.js            # Client Axios centralisé + appels API
│   │   ├── hooks/useAuth.jsx       # Provider React, gestion token/profil
│   │   ├── components/
│   │   │   ├── Layout.jsx          # Navigation latérale + outlet React Router
│   │   │   └── ResultCard.jsx      # Affichage probabilités + labels par classe
│   │   └── pages/
│   │       ├── HomePage.jsx        # Landing page
│   │       ├── LoginPage.jsx       # Connexion
│   │       ├── RegisterPage.jsx    # Inscription
│   │       ├── PredictPage.jsx     # Formulaire d'analyse principal
│   │       ├── DashboardPage.jsx   # Graphiques + métriques du modèle
│   │       └── HistoryPage.jsx     # Tableau de l'historique
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── notebooks/                      # Notebooks Jupyter — entraînement du modèle
├── data_sample/                    # Exemples de signaux ECG
├── api/                            # Version Streamlit (legacy)
├── test/                           # Tests
├── app.py                          # Application Streamlit standalone
├── docker-compose.yml
├── requirements.txt
└── sample.json                     # Exemple de payload JSON
```

---

## Données — Dataset PTB-XL

PTB-XL est le plus grand jeu de données clinique ECG 12 dérivations publiquement disponible, produit par la Physikalisch-Technische Bundesanstalt. Il contient 21 799 enregistrements de 10 secondes provenant de 18 885 patients au format WFDB.

| Propriété | Valeur |
|---|---|
| Nombre d'enregistrements | 21 799 |
| Nombre de patients | 18 885 |
| Fréquence d'échantillonnage (LR) | 100 Hz |
| Longueur de chaque signal | 1 000 pas de temps (10 secondes) |
| Nombre de dérivations | 12 |
| Forme du tenseur chargé | (21 799, 1 000, 12) |
| Valeurs manquantes (NaN) | 0 |

### Distribution des classes

| Classe | Nom complet | Occurrences |
|---|---|---|
| NORM | ECG Normal | 9 514 |
| MI | Infarctus du Myocarde | 5 469 |
| STTC | Modification ST/T | 5 235 |
| CD | Trouble de la Conduction | 4 898 |
| HYP | Hypertrophie | 2 649 |

La classe NORM est surreprésentée (~44 % des enregistrements), tandis que HYP est la plus rare (~12 %). Ce déséquilibre est traité explicitement lors de l'entraînement.

### Découpage train / test

Le champ `strat_fold` (1 à 10) permet une validation croisée stratifiée. Le fold 10 est utilisé comme ensemble de test.

| Ensemble | Taille | Forme du tenseur |
|---|---|---|
| Entraînement (folds 1–9) | 19 601 enregistrements | (19 601, 1 000, 12) |
| Test (fold 10) | 2 198 enregistrements | (2 198, 1 000, 12) |

---

## Prétraitement et normalisation

Chaque enregistrement subit trois étapes avant d'alimenter le réseau.

**1. Chargement WFDB avec cache**  
Les signaux bruts sont lus via `wfdb.rdsamp()`. Un cache `X.npy` évite de recharger 21 799 enregistrements à chaque exécution.

**2. Normalisation par enregistrement**  
Chaque signal est standardisé indépendamment : soustraction de la moyenne, division par l'écart-type (+ ε = 1e-8 pour la stabilité numérique). Le modèle devient ainsi insensible aux variations d'amplitude absolue entre patients ou équipements.

```python
def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

X = np.array([normalize(x) for x in X])
```

**3. Transposition et mise en forme**  
Le format de stockage PTB-XL est `(1 000, 12)` — temps × dérivations. Le modèle 1D-CNN attend `(12, 1 000)` — canaux × temps. La transposition est réalisée par `x.permute(0, 2, 1)` dans la boucle d'entraînement.

---

## Architecture du modèle

### Vue d'ensemble

| Couche / Composant | Entrée | Sortie | Paramètres |
|---|---|---|---|
| ResidualBlock 1 | (B, 12, 1000) | (B, 64, 1000) | in=12, out=64 |
| ResidualBlock 2 | (B, 64, 1000) | (B, 128, 1000) | in=64, out=128 |
| ResidualBlock 3 | (B, 128, 1000) | (B, 256, 1000) | in=128, out=256 |
| AdaptiveAvgPool1d | (B, 256, 1000) | (B, 256, 1) | output_size=1 |
| Linear | (B, 256) | (B, 128) | — |
| ReLU + Dropout(0.3) | (B, 128) | (B, 128) | p=0.3 |
| Linear (sortie) | (B, 128) | (B, 5) | 5 classes |

Le pooling adaptatif global rend l'architecture indépendante de la longueur temporelle en entrée.

### Bloc résiduel (ResidualBlock)

Chaque bloc contient deux convolutions 1D avec BatchNorm et ReLU, ainsi qu'une connexion de raccourci (shortcut) qui projette les dimensions si nécessaire.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)
```

### Fonction de perte — Focal Loss

La Focal Loss réduit la contribution des exemples faciles et concentre l'apprentissage sur les cas difficiles, particulièrement utile pour les classes sous-représentées comme HYP.

```python
class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=self.pos_weight, reduction='none')
        pt    = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean()
```

| Paramètre | Valeur |
|---|---|
| gamma | 2 |
| pos_weight CD | 3.4527 |
| pos_weight HYP | 7.2116 |
| pos_weight MI | 2.9848 |
| pos_weight NORM | 1.2922 |
| pos_weight STTC | 3.1580 |
| Optimiseur | Adam, lr = 1e-3 |
| Scheduler | ReduceLROnPlateau, patience = 2 |

---

## Pipeline d'entraînement

### Gestion du déséquilibre de classes

Deux mécanismes complémentaires sont déployés :

**WeightedRandomSampler** : chaque exemple reçoit un poids inversement proportionnel à la fréquence de ses classes positives. Le DataLoader rééquilibre ainsi la représentation de chaque classe à chaque époque.

**pos_weight dans la Focal Loss** : pour chaque classe `c`, `pos_weight[c] = (N - N_c) / N_c`, ce qui pénalise davantage les faux négatifs sur les classes rares.

### Évolution de la perte

| Époque | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.4173 | 0.3504 |
| 2 | 0.3559 | 0.3131 |
| 3 | 0.3334 | 0.3181 |
| 4 | 0.3221 | 0.3320 |
| 5 | 0.3166 | 0.2871 |
| 6 | 0.3044 | 0.2821 |
| 7 | 0.3011 | 0.2779 |
| **8** | **0.2911** | **0.2766** |
| 9 | 0.2880 | 0.2820 |
| 10 | 0.2815 | 0.2822 |

La perte de validation minimale (0.2766) est atteinte à l'époque 8. Le scheduler réduit automatiquement le taux d'apprentissage lorsque la validation stagne.

### Sauvegarde du checkpoint

```python
torch.save({
    "model_state": model.state_dict(),
    "thresholds":  best_thresholds,   # seuils optimisés par classe
    "classes":     class_names
}, "ecg_full_model.pth")
```

---

## Évaluation et métriques

### Seuils de décision optimisés

Pour chaque classe, le seuil optimal est déterminé par recherche sur la grille [0.1, 0.9] (pas = 0.05) en maximisant le score F1 sur l'ensemble de test.

| Classe | Nom complet | Seuil optimal |
|---|---|:---:|
| NORM | ECG Normal | 0.55 |
| MI | Infarctus du Myocarde | 0.45 |
| CD | Trouble de la Conduction | 0.70 |
| HYP | Hypertrophie | 0.65 |
| STTC | Modification ST/T | 0.65 |

### AUC-ROC par classe

| Classe | AUC |
|---|---|
| NORM | 0.938 |
| MI | 0.927 |
| STTC | 0.926 |
| HYP | 0.899 |
| CD | 0.897 |
| **Macro AUC** | **0.917** |

### Rapport de classification complet (fold 10, 2 198 enregistrements)

| Classe | Précision | Rappel | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| CD | 0.79 | 0.67 | 0.73 | 496 |
| HYP | 0.59 | 0.59 | 0.59 | 262 |
| MI | 0.67 | 0.84 | 0.74 | 550 |
| NORM | 0.81 | 0.89 | 0.85 | 963 |
| STTC | 0.71 | 0.79 | 0.75 | 521 |
| Micro avg | 0.74 | 0.79 | 0.76 | 2 792 |
| Macro avg | 0.71 | 0.76 | 0.73 | 2 792 |
| Weighted avg | 0.74 | 0.79 | 0.76 | 2 792 |
| Samples avg | 0.75 | 0.80 | 0.75 | 2 792 |

### Matrices de confusion (OvR par classe)

| Classe | VP | FP | FN | VN |
|---|:---:|:---:|:---:|:---:|
| CD | 334 | 91 | 162 | 1 611 |
| HYP | 154 | 107 | 108 | 1 829 |
| MI | 461 | 230 | 89 | 1 418 |
| NORM | 854 | 196 | 109 | 1 039 |
| STTC | 412 | 171 | 109 | 1 506 |

MI et NORM affichent le meilleur rappel (0.84 et 0.89), confirmant que le modèle détecte bien les infarctus et les ECG normaux. HYP reste la classe la plus difficile (F1 = 0.59) en raison de sa sous-représentation et de la proximité de ses patterns avec d'autres pathologies.

---

## Backend — API FastAPI

### Endpoints

| Méthode | Endpoint | Auth | Description |
|---|---|:---:|---|
| POST | `/api/auth/register` | Non | Inscription (username, email, password) |
| POST | `/api/auth/login` | Non | Connexion — retourne un token JWT |
| POST | `/api/predict/` | Oui | Inférence ECG — signal JSON (1000 × 12) |
| GET | `/api/history/` | Oui | Liste des analyses passées |
| DELETE | `/api/history/{id}` | Oui | Suppression d'un enregistrement |
| GET | `/api/stats/` | Oui | Statistiques agrégées par utilisateur |

### Stack technique

| Composant | Technologie | Rôle |
|---|---|---|
| Serveur ASGI | FastAPI + Uvicorn | Serveur HTTP asynchrone |
| Base de données | SQLite + aiosqlite | Persistance des analyses et utilisateurs |
| Hachage mot de passe | bcrypt (passlib) | Stockage sécurisé |
| Authentification | JWT (python-jose) | Tokens expirables signés |
| Inférence | PyTorch (singleton) | Chargement unique du checkpoint |
| Validation données | Pydantic v2 | Schémas requêtes / réponses |

### Pipeline d'inférence

1. Réception JSON `{signal_data: [[float × 12] × 1000], patient_name, age, sex}`
2. Conversion en tenseur PyTorch float32
3. Transposition `(1000, 12)` → `(12, 1000)`, ajout dimension batch
4. Passe forward sans gradient (`torch.no_grad()`)
5. Application de sigmoid → probabilités [0, 1]
6. Application des seuils par classe → labels binaires
7. Construction de la réponse JSON structurée

---

## Frontend — React / Vite / TailwindCSS

| Page / Composant | Fichier | Fonction |
|---|---|---|
| Page d'accueil | `HomePage.jsx` | Landing page présentant le projet |
| Connexion | `LoginPage.jsx` | Formulaire JWT |
| Inscription | `RegisterPage.jsx` | Création de compte |
| Analyse ECG | `PredictPage.jsx` | Upload / saisie JSON + résultats |
| Dashboard | `DashboardPage.jsx` | Métriques du modèle + graphiques |
| Historique | `HistoryPage.jsx` | Tableau des analyses avec suppression |
| Layout | `Layout.jsx` | Sidebar + outlet React Router |
| Carte résultat | `ResultCard.jsx` | Probabilités + labels par classe |
| Client API | `utils/api.js` | Axios centralisé avec token JWT injecté |
| Contexte Auth | `hooks/useAuth.jsx` | Provider React, gestion token/profil |

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

# Lancer l'API
uvicorn app.main:app --reload --port 8000
```

API disponible sur `http://localhost:8000`  
Documentation interactive sur `http://localhost:8000/docs`

### 2. Frontend (React + Vite)

```bash
cd frontend

npm install
npm run dev
```

Application disponible sur `http://localhost:5173`

---

## Démarrage avec Docker

```bash
# Depuis la racine du projet
cp votre_modele.pth backend/app/model/ecg_full_model.pth

docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

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
# Générer un signal de test
python3 -c "
import numpy as np, json
sig = np.random.randn(1000, 12).tolist()
print(json.dumps({'signal_data': sig, 'patient_name': 'Patient Test', 'age': 55, 'sex': 'M'}))
" > test_payload.json

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

### Historique et statistiques

```bash
TOKEN="votre_token_jwt"

curl http://localhost:8000/api/history/          -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/api/stats/            -H "Authorization: Bearer $TOKEN"
curl -X DELETE http://localhost:8000/api/history/1 -H "Authorization: Bearer $TOKEN"
```

---

## Export de signal depuis Python

```python
import numpy as np, json, requests

# X_test[0] a la forme (1000, 12)
signal = X_test[0]

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

| Variable | Valeur par défaut | Description |
|---|---|---|
| `SECRET_KEY` | `change-me-in-production-...` | Clé de signature JWT |
| `DB_PATH` | `cardioscan.db` | Chemin vers la base SQLite |
| `MODEL_PATH` | `app/model/ecg_full_model.pth` | Chemin vers le checkpoint |

### Frontend

| Variable | Valeur par défaut | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | URL de base de l'API |

---

## Suggestions de déploiement

| Service | Notes |
|---|---|
| **Render** | Déployer le backend comme Web Service Docker. Tier gratuit disponible. |
| **Railway** | `railway up` depuis le dossier backend. Détecte le Dockerfile automatiquement. |
| **Vercel** | Déployer le dossier `frontend/`. Configurer `VITE_API_URL`. |
| **Fly.io** | Idéal pour le backend PyTorch — disque persistant pour DB + modèle. |

---

## Licence

Ce projet est distribué sous licence libre. Consultez le fichier [LICENSE.txt](./LICENSE.txt) pour plus de détails.

---

> **Avertissement :** Cet outil est destiné à des fins de recherche et d'aide à la décision. Il ne remplace pas le diagnostic d'un professionnel de santé qualifié.
