# SCANIA Component X — Case Study 2

## Dataset

Le PDF décrit bien le **SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance** disponible sur [Kaggle](https://www.kaggle.com/datasets/mariembenkamel/scania-component-x-dataset/data).

Le script utilise uniquement la bibliothèque standard Python : ni pandas, ni `kagglehub`, et donc aucun cache externe. L'archive est téléchargée directement depuis l'API Kaggle, extraite dans `data/` à la racine du projet, puis supprimée. Le dossier `data/` est ignoré par Git.

Prérequis : Python 3.13 ou plus récent.

```bash
python download_dataset.py
```

La même commande fonctionne sous macOS, Linux et Windows. Sous Windows, `python` peut être remplacé par `py` :

```powershell
py download_dataset.py
```

Options utiles :

```bash
# Retélécharger et remplacer les fichiers existants
python download_dataset.py --force

# Conserver aussi l'archive ZIP dans data/
python download_dataset.py --keep-archive

# Choisir un autre dossier
python download_dataset.py --output chemin/vers/data
```

Le téléchargement représente environ **1,54 GiB**. Le dataset Kaggle étant public, aucune authentification ne devrait être nécessaire. Si Kaggle l'exige, définir `KAGGLE_USERNAME` et `KAGGLE_KEY` dans l'environnement avant d'exécuter le script.
