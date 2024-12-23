# TP : Classification des Fleurs Iris avec KNN

Ce projet a été réalisé dans le cadre d’un TP data science . L’objectif principal était de comprendre et de mettre en œuvre l’algorithme **K-Nearest Neighbors (KNN)** pour résoudre un problème de classification.

## **Contexte**

L’objectif du TP était de :
1. Explorer un dataset classique, le **dataset Iris**, utilisé pour l’apprentissage des bases du machine learning.
2. Implémenter l’algorithme KNN et comprendre son fonctionnement.
3. Appliquer les étapes fondamentales d’un workflow machine learning :
   - Chargement des données.
   - Prétraitement.
   - Entraînement et évaluation d’un modèle.


## **Description du Dataset**

Le dataset utilisé, appelé **Iris dataset**, est intégré à la bibliothèque `sklearn`. Il contient :
- **150 exemples** répartis en 3 classes : 
  - *Setosa*, *Versicolor*, *Virginica*.
- **4 caractéristiques** :
  - Longueur et largeur du sépale (cm).
  - Longueur et largeur du pétale (cm).
- **1 variable cible** : La classe de la fleur.


## **Étapes Réalisées**

### **1. Exploration des Données**
Nous avons chargé et exploré les données pour identifier leur structure et vérifier l’absence de valeurs manquantes.


### **2. Visualisation des Données**
Des visualisations ont été réalisées pour mieux comprendre les relations entre les caractéristiques et les classes cibles.


### **3. Entraînement du Modèle**
- Division du dataset en deux ensembles : **Entraînement (80%)** et **Test (20%)**.
- Entraînement de l’algorithme KNN avec un paramètre `k=3`.

### **4. Évaluation du Modèle**
L’évaluation du modèle a été effectuée à l’aide de métriques telles que l’accuracy et la matrice de confusion.


## **Résultats**

- **Accuracy obtenue** : Environ 95%.
- **Observations** :
  - Les caractéristiques liées aux pétales (longueur et largeur) ont un fort pouvoir discriminant pour classer les fleurs.
  - L’algorithme KNN est performant pour ce dataset, mais il est sensible au choix de `k` et à la distribution des données.

## **Technologies Utilisées**

- **Langage** : Python
- **Bibliothèques principales** :
  - `pandas` : Pour la manipulation des données.
  - `numpy` : Pour les calculs numériques.
  - `scikit-learn` : Pour l’implémentation du modèle KNN et l’évaluation.
  - `seaborn` et `matplotlib` : Pour les visualisations.

## **Comment Exécuter le Projet**

1. Clone ce dépôt :
   ```bash
   git clone https://github.com/sihamaitbaziz/IRIS-KNN.git
2.Navigue dans le dossier du projet :
```bash
   cd IRIS-KNN
```
3.Installe les dépendances :
```bash
   pip install -r requirements.txt
```
4.Exécute le script :
```bash
  python iris.py
```


