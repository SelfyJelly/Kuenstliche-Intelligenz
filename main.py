# Bibliotheken

# Pandas
import pandas as pd 

# NumPy
import numpy as np

# Matplotlib
import matplotlib 
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import f1_score as f1, precision_score, recall_score

# Pfad zur Datei 
data = pd.read_csv("speeddating.csv")

# automatisch ermittelte Datentypen der Datei prüfen:
print(data.dtypes)
print(data.head())

# Übersicht über Anzahl an Zeilen und Spalten des Datensatzes und über die vorhandenen Typen
print(data.info())

# Übersicht über die deskriptive Statistik der Spalten
print(data.describe())

# Information für eine bestimmte Spalte - hier die Information, ob beide Beteiligten sich wiedersehen wollen
print(data['match'].describe())

# in scikit_learn steht X für die Attribute und Y für die Klassen. Wir teilen unsere Daten entsprechend auf:

# [:,0:122] bedeutet "alle Zeilen für die Spalten 0-121 (alle 122 Attribute)
X = data.iloc[:,0:122]
X = X.drop(columns=["decision", "decision_o"])

# ... oder wir wählen die gewünschte Spalte mit Namen aus 
# (und wandeln sie in ein eindimensionales Array um, wie es der Lerner erwartet)
y = data[["match"]].values.ravel()

# 20% der Daten als Testdaten zurückbehalten
# Setzen des random_state ermöglicht ein Reproduzieren der Aufteilung; der konkrete Wert ist egal, solange es immer derselbe ist
# stratify bedeutet, dass wir in den Test- und Trainingsdaten dasselbe Verhältnis von True und False in der Zielklasse haben möchten
# (daher geben wir die Zielklassenverteilung als Argument mit)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

# Hat alles funktioniert? shape gibt (Zeilen, Spalten) des Data Frames zurück
print(data.shape)
print(X.shape)
print(y.shape)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print()
# Der Lerner, den wir nachher benutzen werden, unterstützt keine category-Features (wie "female"/"male"). 
# Wir filtern diese Features daher aus.
# Die Boole'schen Features werden wieder in 1.0 und 0.0 umgewandelt - nicht ganz sauber, aber effektiv.

X_train = X_train.select_dtypes(include=[np.number,bool])
X_test = X_test.select_dtypes(include=[np.number,bool])

# Fehlende Daten (s. "has_missing") sind kein Problem: https://scikit-learn.org/stable/modules/tree.html#tree-algorithms

# Wie viele Spalten sind übrig? 
print(X_test.shape)

# Algorithmus auswählen und untrainiertes Modell erstellen:

clf = tree.DecisionTreeClassifier(random_state=42)

# Modell trainieren:
clf = clf.fit(X_train, y_train)

# Wie gut ist das Modell?

# Vorhersage für die Testdaten machen und in "predict" speichern 
# -- jetzt gibt es für jeden Datenpunkt in den Testdaten eine Modellvorhersage
# Die Zielklasse der Testdaten geht natürlich nicht ein, nur die Features:
predict = clf.predict(X_test)

#Standard-Evaluationsmaße berechnen: Passen die Vorhersagen in predict zu den echten Zielklassen in y_test?
print("F1: ",f1(predict, y_test))
print("Precision: ",precision_score(y_test, predict))
print("Recall: ",recall_score(y_test, predict))
# Schauen wir uns den Baum an (wir geben die Spaltennamen aus den Trainingsdaten mit, 
# damit die Features für uns leichter zu identifizieren sind): 

tree.plot_tree(clf, feature_names=list(X_train.columns))


