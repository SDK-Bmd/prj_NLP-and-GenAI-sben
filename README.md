# Projets d'Analyse de Données Amazon

## 🚀 Introduction

Ce dépôt contient deux projets complémentaires d'analyse de données Amazon:

### Projet 1 : Topic Modeling des Avis des Produits
Application d'analyse des avis clients utilisant des techniques de Topic Modeling et d'analyse de sentiment pour extraire des insights significatifs des commentaires utilisateurs.

### Projet 2 : Retrieval Augmented Generation (RAG)
Implémentation d'une architecture RAG pour analyser et interroger intelligemment les descriptions de produits Amazon.

## 🛠 Technologies Utilisées

### Technologies Communes
- **Python** : Langage de programmation principal
- **Streamlit** : Interface utilisateur interactive
- **Sentence Transformers** : Génération d'embeddings vectoriels

### Spécifique au Projet 1
- **spaCy** : Traitement du langage naturel
- **scikit-learn** : Clustering et vectorisation
- **BERT** : Analyse de sentiment
- **Matplotlib & Seaborn** : Visualisation des données

### Spécifique au Projet 2
- **Weaviate** : Base de données vectorielle
- **OpenAI GPT-3.5** : Génération de réponses
- **Multiprocessing** : Traitement parallèle

## 📦 Prérequis

### Configuration Système
- Python 3.8+
- Git

### Dépendances Communes
```bash
pip install streamlit sentence-transformers pandas numpy
```

### Projet 1
```bash
pip install spacy scikit-learn torch transformers matplotlib seaborn
python -m spacy download en_core_web_sm
```

### Projet 2
```bash
pip install weaviate-client openai python-dotenv
```


## 🔧 Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/SDK-Bmd/prj_NLP-and-GenAI-sben.git
cd amazon-analysis-projects
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configuration spécifique au Projet 2 :
Créez un fichier `.env` :
```
OPENAI_API_KEY=votre-cle-openai
WEAVIATE_URL=http://localhost:8080
```

### Installation de Weaviate (Projet 2)

#### Option 1 : Installation avec Docker (Recommandée)

1. Installez Docker sur votre système si ce n'est pas déjà fait

2. Exécutez Weaviate via Docker Compose :
```bash
# Créez un fichier docker-compose.yml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - 8080:8080
    environment:
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      DRONE: 'true'
      OPENAI_APIKEY: ${OPENAI_API_KEY}

# Lancez le service
docker-compose up -d
```

Alternative pour Windows avec Docker Desktop :
```bash
docker run -d -p 8080:8080 semitechnologies/weaviate
```

3. Vérifiez l'installation :
```bash
curl http://localhost:8080/v1/.well-known/ready
```

#### Option 2 : Installation Locale

1. Installez Python 3.8+ et pip

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
```

3. Installez Weaviate Client :
```bash
pip install weaviate-client
```

4. Installez une instance locale de Weaviate :
```bash
# Sous Linux/macOS
pip install weaviate-local

# Démarrez le service local
weaviate-local start
```

## 🚀 Utilisation

### Projet 1 : Topic Modeling
```bash
streamlit run nlp_streamlit.py
```

#### Fonctionnalités :
- Chargement des fichiers JSONL (reviews.jsonl et meta.jsonl)
- Prétraitement et clustering des documents
- Reconnaissance d'entités nommées
- Analyse de sentiment avec visualisations

### Projet 2 : RAG
```bash
streamlit run archi_REG.py
```

#### Fonctionnalités :
- Chargement des descriptions de produits
- Recherche sémantique
- Génération de réponses contextuelles
- Paramétrage de la créativité des réponses

## 📊 Processus de Traitement

### Projet 1
1. **Prétraitement** : 
   - Tokenisation et lemmatisation
   - Extraction de bigrammes
   - Vectorisation (TF-IDF ou embeddings)
2. **Analyse** :
   - Clustering (KMeans/DBSCAN)
   - Analyse de sentiment
3. **Visualisation** :
   - Graphiques de clusters
   - Distribution des sentiments
   - Statistiques d'entités

### Projet 2
1. **Traitement Initial** :
   - Fractionnement des textes
   - Génération d'embeddings
   - Stockage vectoriel
2. **Recherche et Génération** :
   - Recherche sémantique
   - Génération de réponses
   - Optimisation contextuelle

## 🛡 Sécurité et Optimisation

- Gestion sécurisée des clés API
- Mise en cache des calculs intensifs
- Validation des entrées utilisateur
- Gestion efficace de la mémoire

## 🔜 Développements Futurs

- Support multilingue étendu
- Intégration de nouveaux modèles d'embedding
- Amélioration des visualisations
- Export des résultats
- Personnalisation avancée des paramètres

## 👥 Contributeurs

- Sedik BENMESSAOUD

## 🐛 Support

Pour signaler un bug ou proposer une amélioration, veuillez utiliser la section "Issues" du dépôt GitHub.