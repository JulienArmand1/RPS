import bibtexparser
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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

# --- 2. Graphique : 10 auteurs les plus populaires ---

df['Auteurs'] = df['Auteurs'].fillna('')
all_authors = []
for authors in df['Auteurs']:
    # On suppose que les auteurs sont séparés par " and " dans le BibTeX
    splitted = [a.strip() for a in authors.split(' and ')]
    all_authors.extend([a for a in splitted if a])

author_counts = Counter(all_authors)
author_df = pd.DataFrame(author_counts.items(), columns=['Auteur', 'Nombre'])
author_df = author_df.sort_values('Nombre', ascending=False)

# On ne conserve que les 10 plus populaires
top_10_authors = author_df.head(10)

plt.figure(figsize=(10, 6))
plt.bar(top_10_authors['Auteur'], top_10_authors['Nombre'])
plt.xlabel("Auteurs")
plt.ylabel("Nombre d'apparitions")
plt.title("Top 10 des auteurs les plus populaires")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# --- 3. Graphique : 10 conférences/journaux les plus populaires ---

df['Journal/Conférence'] = df['Journal/Conférence'].fillna('')
journal_counts = df['Journal/Conférence'].value_counts()

# On ne conserve que les 10 plus populaires
top_10_journals = journal_counts.head(10)

plt.figure(figsize=(10, 6))
plt.bar(top_10_journals.index, top_10_journals.values)
plt.xlabel("Journal / Conférence")
plt.ylabel("Nombre d'articles")
plt.title("Top 10 des conférences/journaux les plus populaires")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
