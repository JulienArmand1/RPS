import bibtexparser
import xml.etree.ElementTree as ET

def create_simple_mindmap(bib_file_path, output_mm_path):
    # Lecture et parsing du fichier BibTeX
    with open(bib_file_path, 'r', encoding='utf-8') as bib_file:
        bib_database = bibtexparser.load(bib_file)
    
    # Création de l'élément racine pour Freemind
    map_element = ET.Element('map')
    map_element.set('version', '0.9.0')
    
    # Nœud racine de la mind map
    root_node = ET.SubElement(map_element, 'node')
    root_node.set('TEXT', 'Articles')
    
    # Pour chaque article, créer un nœud avec son titre et ajouter ses mots-clés en sous-nœuds
    for entry in bib_database.entries:
        # Titre de l'article (ou valeur par défaut)
        title = entry.get('title', 'Sans titre')
        article_node = ET.SubElement(root_node, 'node')
        article_node.set('TEXT', title)
        
        # Si des mots-clés sont définis, les ajouter comme sous-nœuds de l'article
        keywords = entry.get('keywords', '').strip()
        if keywords:
            # On suppose que les mots-clés sont séparés par des virgules
            for keyword in [k.strip() for k in keywords.split(',')]:
                kw_node = ET.SubElement(article_node, 'node')
                kw_node.set('TEXT', keyword)
    
    # Sauvegarde du fichier XML en format Freemind (.mm)
    tree = ET.ElementTree(map_element)
    tree.write(output_mm_path, encoding='utf-8', xml_declaration=True)
    print(f"Mindmap générée dans : {output_mm_path}")

if __name__ == "__main__":
    # Remplacez "votre_fichier.bib" par le chemin de votre fichier BibTeX et "mindmap.mm" par le nom désiré
    create_simple_mindmap("references.bib", "mindmap.mm")
