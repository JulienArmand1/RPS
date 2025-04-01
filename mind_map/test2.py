import bibtexparser
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from venn import venn  # pip install venn

# --- 1. Lecture du fichier BibTeX et création du DataFrame ---
with open('references.bib', encoding='utf-8') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

data = []
for entry in bib_database.entries:
    data.append({
        'Type': entry.get('ENTRYTYPE', ''),
        'Clé': entry.get('ID', ''),
        'Titre': entry.get('title', ''),
        'Auteurs': entry.get('author', ''),
        'Année': entry.get('year', ''),
        'Tags': entry.get('keywords', ''),
        'Journal/Conférence': entry.get('journal', '') or entry.get('booktitle', '')
    })

df = pd.DataFrame(data)
total_docs = len(df)
print("Nombre total de documents :", total_docs)

# --- 2. Graphique de la répartition par année ---
df['Année'] = pd.to_numeric(df['Année'], errors='coerce')
year_counts = df['Année'].value_counts().sort_index()

plt.figure(figsize=(10,6))
plt.bar(year_counts.index.dropna().astype(int), year_counts.dropna().values)
plt.xlabel('Année')
plt.ylabel("Nombre de documents")
plt.title("Répartition des documents par année")
plt.xticks(year_counts.index.dropna().astype(int))
plt.tight_layout()
plt.show()

# --- 3. Construction des ensembles (sets) pour le diagramme de Venn ---
# Tags que l'on veut inclure dans le diagramme
tags_of_interest = ["multi-agent", "multi-player", "evolutionnary game", "game theory"]

# Dictionnaire { "multi-agent": set_des_clés, "multi-player": set_des_clés, ... }
venn_sets = {tag: set() for tag in tags_of_interest}

# Parcourt chaque ligne du DataFrame et ajoute la clé aux sets appropriés
for i, row in df.iterrows():
    tag_str = row['Tags']
    if pd.notnull(tag_str):
        tags_list = [t.strip().lower() for t in tag_str.split(',')]
        for tag in tags_of_interest:
            if tag in tags_list:
                venn_sets[tag].add(row['Clé'])

# --- 4. Affichage des counts par intersection (facultatif) ---
# On peut compter manuellement le nombre de documents dans chaque intersection.
# C'est purement informatif pour la console :
print("\nRépartition par intersections de tags :")
counter = Counter()
for i, row in df.iterrows():
    if pd.notnull(row['Tags']):
        tags_list = [t.strip().lower() for t in row['Tags'].split(',')]
        present = set(tag for tag in tags_of_interest if tag in tags_list)
        counter[frozenset(present)] += 1

# On affiche seulement les ensembles non vides
for subset, count in sorted(counter.items(), key=lambda x: -x[1]):
    if subset:  # on ignore le frozenset() vide
        print(f"{', '.join(subset)} : {count}")

# --- 5. Création du diagramme de Venn ---
plt.figure(figsize=(10,10))
venn(venn_sets)
plt.title("Diagramme de Venn pour 4 tags")
plt.tight_layout()
plt.show()
