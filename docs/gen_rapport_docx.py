"""
gen_rapport_docx.py
-------------------
Génère le rapport complet du projet Boeuf Tracker au format Word (.docx).
- Aucun nom d'auteur (laissé vide)
- Aucun diagramme ASCII : utilise les 6 PNG de docs/diagram_*.png
- Thème professionnel : navy/green, Calibri
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
OUT = os.path.join(ROOT, "Boeuf_Tracker_Rapport_Complet.docx")

# Palette
NAVY = RGBColor(0x00, 0x33, 0x66)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
RED = RGBColor(0xCC, 0x00, 0x00)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x60, 0x60, 0x60)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = "EAF2F8"
NAVY_HEX = "003366"
GREEN_HEX = "2E8B57"

# ============================================================================
# Helpers
# ============================================================================

def set_cell_bg(cell, color_hex):
    """Couleur de fond d'une cellule."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    """Marges internes d'une cellule (en twips)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    m = OxmlElement("w:tcMar")
    for tag, val in (("top", top), ("bottom", bottom), ("start", left), ("end", right)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        m.append(el)
    tc_pr.append(m)


def add_page_break(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)


def style_run(run, size=11, bold=False, italic=False, color=DARK, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def add_heading_styled(doc, text, level=1):
    """Titre avec couleur et style personnalisé."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    sizes = {1: 20, 2: 15, 3: 12.5}
    colors = {1: NAVY, 2: GREEN, 3: NAVY}
    run = p.add_run(text)
    style_run(run, size=sizes.get(level, 11), bold=True, color=colors.get(level, NAVY))
    # Marque comme heading pour navigation Word
    p.style = doc.styles[f"Heading {level}"]
    # Re-applique couleur après style
    for r in p.runs:
        style_run(r, size=sizes.get(level, 11), bold=True, color=colors.get(level, NAVY))
    return p


def add_para(doc, text, size=11, bold=False, italic=False, color=DARK, align=None,
             space_after=6, indent=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.3
    if align is not None:
        p.alignment = align
    if indent is not None:
        p.paragraph_format.left_indent = Cm(indent)
    run = p.add_run(text)
    r = run
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return p


def add_bullet(doc, text, level=0, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.left_indent = Cm(0.75 + level * 0.5)
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        style_run(r1, size=11, bold=True, color=NAVY)
        r2 = p.add_run(text)
        style_run(r2, size=11)
    else:
        r = p.add_run(text)
        style_run(r, size=11)
    return p


def add_image(doc, img_name, width_inches=6.0, caption=None):
    """Insère une image centrée + légende optionnelle."""
    path = os.path.join(DOCS, img_name)
    if not os.path.exists(path):
        add_para(doc, f"[Image manquante: {img_name}]", italic=True, color=GREY)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run()
    run.add_picture(path, width=Inches(width_inches))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        cr = cap.add_run(caption)
        style_run(cr, size=9.5, italic=True, color=GREY)


def add_table(doc, headers, rows, col_widths=None, header_bg=NAVY_HEX,
              alt_bg=LIGHT_BG, first_col_bold=False):
    """Tableau stylisé avec en-tête coloré et lignes alternées."""
    n_cols = len(headers)
    table = doc.add_table(rows=1, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # En-tête
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        set_cell_bg(cell, header_bg)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        style_run(r, size=10.5, bold=True, color=WHITE)

    # Lignes
    for ridx, row_data in enumerate(rows):
        row = table.add_row()
        bg = alt_bg if ridx % 2 == 0 else "FFFFFF"
        for i, val in enumerate(row_data):
            cell = row.cells[i]
            cell.text = ""
            set_cell_bg(cell, bg)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            style_run(r, size=10, bold=(first_col_bold and i == 0),
                      color=NAVY if (first_col_bold and i == 0) else DARK)

    # Largeurs
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table


def add_callout(doc, title, text, color_hex=GREEN_HEX, icon="➜"):
    """Encadré coloré pour notes/astuces."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.text = ""
    set_cell_bg(cell, "F0F7F4")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    # Bordure gauche colorée
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:color"), color_hex)
    borders.append(left)
    tc_pr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r1 = p.add_run(f"{icon} {title}")
    style_run(r1, size=11, bold=True, color=RGBColor.from_string(color_hex))
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    r2 = p2.add_run(text)
    style_run(r2, size=10.5, color=DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_code_block(doc, code, language="python"):
    """Bloc de code avec fond gris et police monospace."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.text = ""
    set_cell_bg(cell, "F5F5F5")
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
    for line in code.strip("\n").split("\n"):
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(line if line else " ")
        r.font.size = Pt(9.5)
        r.font.name = "Consolas"
        r.font.color.rgb = RGBColor(0x2D, 0x2D, 0x2D)
    # Supprimer le premier paragraphe vide
    first = cell.paragraphs[0]
    first._element.getparent().remove(first._element)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ============================================================================
# Configuration du document
# ============================================================================

def configure_document(doc):
    """Styles globaux, marges, police par défaut."""
    # Marges
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.4)
        section.right_margin = Cm(2.4)

    # Police par défaut
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = DARK
    style.paragraph_format.line_spacing = 1.3
    style.paragraph_format.space_after = Pt(6)


# ============================================================================
# PAGE DE COUVERTURE
# ============================================================================

def build_cover(doc):
    # Espace haut
    for _ in range(4):
        doc.add_paragraph()

    # Logo / titre principal
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("BOEUF TRACKER")
    style_run(r, size=42, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Système de Surveillance et d'Identification")
    style_run(r, size=16, color=GREEN)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(20)
    r = p.add_run("de Bovins par Vision par Ordinateur")
    style_run(r, size=16, color=GREEN)

    # Ligne décorative
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("________________________________________")
    style_run(r, size=12, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(40)
    r = p.add_run("Rapport Technique Complet")
    style_run(r, size=18, bold=True, color=DARK)

    # Tableau d'information
    table = doc.add_table(rows=5, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    infos = [
        ("Projet", "Boeuf Tracker v2.0.0"),
        ("Type", "PFE Individuel — Projet de Fin d'Études"),
        ("Cours", "GEI1052 — Génie Électrique"),
        ("Session", "Hiver 2026"),
        ("Auteur", ""),
    ]
    for i, (k, v) in enumerate(infos):
        row = table.rows[i]
        row.height = Cm(0.9)
        c1, c2 = row.cells
        c1.text = ""
        c2.text = ""
        set_cell_bg(c1, NAVY_HEX)
        set_cell_margins(c1)
        set_cell_margins(c2)
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c2.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        r1 = c1.paragraphs[0].add_run(k)
        style_run(r1, size=11, bold=True, color=WHITE)
        r2 = c2.paragraphs[0].add_run(v)
        style_run(r2, size=11)
        c1.width = Inches(1.8)
        c2.width = Inches(4.0)

    add_page_break(doc)


# ============================================================================
# TABLE DES MATIÈRES
# ============================================================================

def build_toc(doc):
    add_heading_styled(doc, "Table des matières", level=1)
    toc_items = [
        ("1.  Résumé exécutif", "3"),
        ("2.  Contexte et problématique", "4"),
        ("3.  Objectifs du projet", "6"),
        ("4.  État de l'art", "7"),
        ("5.  Architecture globale du système", "9"),
        ("6.  Détection YOLO et segmentation", "12"),
        ("7.  Suivi multi-objets (tracking)", "15"),
        ("8.  Ré-identification (Re-ID) avec DINOv2", "17"),
        ("9.  Classification de race avec SigLIP-2", "21"),
        ("10. Génération de noms et persistance", "24"),
        ("11. Pipeline de traitement vidéo", "26"),
        ("12. Classification des comportements", "29"),
        ("13. Analyse de données et tableau de bord", "31"),
        ("14. Carte de chaleur spatiale (heatmap)", "33"),
        ("15. Serveur Flask et API REST", "35"),
        ("16. Interface utilisateur web", "38"),
        ("17. Application desktop Tauri", "41"),
        ("18. Optimisations de performance", "44"),
        ("19. Robustesse et gestion d'erreurs", "47"),
        ("20. Évolution chronologique du projet", "49"),
        ("21. Résultats et métriques", "51"),
        ("22. Limites connues", "54"),
        ("23. Perspectives et améliorations futures", "56"),
        ("24. Guide d'installation et de déploiement", "58"),
        ("25. Structure du code source", "61"),
        ("26. Glossaire technique", "63"),
        ("27. Références", "65"),
        ("    Conclusion", "67"),
    ]
    table = doc.add_table(rows=len(toc_items), cols=2)
    for i, (title, page) in enumerate(toc_items):
        c1, c2 = table.rows[i].cells
        c1.text = ""
        c2.text = ""
        set_cell_margins(c1, top=40, bottom=40)
        set_cell_margins(c2, top=40, bottom=40)
        r1 = c1.paragraphs[0].add_run(title)
        style_run(r1, size=11, bold="." not in title[:3] and title[0].isdigit())
        c2.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r2 = c2.paragraphs[0].add_run(page)
        style_run(r2, size=11, color=GREY)
        c1.width = Inches(5.5)
        c2.width = Inches(0.8)
    add_page_break(doc)


# ============================================================================
# SECTIONS DU RAPPORT
# ============================================================================

def section_resume(doc):
    add_heading_styled(doc, "1. Résumé exécutif", level=1)

    add_para(doc,
        "Boeuf Tracker est un système complet de surveillance d'élevage bovin reposant sur "
        "la vision par ordinateur et l'intelligence artificielle. Conçu dans le cadre d'un "
        "Projet de Fin d'Études en Génie Électrique (GEI1052, Hiver 2026), il permet de "
        "détecter, identifier individuellement, classer par race et analyser spatialement "
        "les bovins présents dans un flux vidéo en temps réel.")

    add_para(doc, "Le système combine quatre technologies d'IA de pointe :")

    add_bullet(doc, "détection et segmentation d'instances en temps réel (~12 ms par frame)", bold_prefix="YOLOv26 sur MLX (Metal) — ")
    add_bullet(doc, "ré-identification cross-vidéo via embeddings de 464 dimensions (~5 ms par batch asynchrone)", bold_prefix="DINOv2-small — ")
    add_bullet(doc, "classification zero-shot de 10 races bovines françaises (~22 ms, nouveau bovin seulement)", bold_prefix="SigLIP-2 So400m — ")
    add_bullet(doc, "classification de comportements (couché, pâture, marche, etc.)", bold_prefix="Numba JIT — ")

    add_heading_styled(doc, "Performances clés", level=3)
    add_table(doc,
        ["Métrique", "Valeur", "Objectif", "Statut"],
        [
            ["FPS moyen (Apple M1 Pro)", "23–26 FPS", "15 FPS", "Dépassé"],
            ["Précision de détection", "~92 %", "—", "Atteint"],
            ["Objectifs livrés", "16 / 16", "6", "Dépassé"],
            ["Plateformes supportées", "macOS, Windows, Linux", "1", "Dépassé"],
            ["Lignes de code", "~8 000", "—", "—"],
        ],
        col_widths=[2.2, 1.6, 1.3, 1.1])

    add_heading_styled(doc, "Composants du système", level=3)
    add_image(doc, "diagram_architecture.png", width_inches=6.2,
              caption="Figure 1 — Architecture globale du système (deux processus : wrapper Tauri + worker Python)")


def section_contexte(doc):
    add_heading_styled(doc, "2. Contexte et problématique", level=1)

    add_heading_styled(doc, "2.1 Le défi de la surveillance d'élevage", level=2)
    add_para(doc,
        "L'élevage bovin moderne représente un secteur économique majeur, particulièrement "
        "en France où l'on compte environ 19 millions de bovins répartis sur quelque "
        "180 000 exploitations. La surveillance manuelle de tels cheptels est :")
    add_bullet(doc, "en main-d'œuvre (2–3 heures par jour par exploitation pour le seul comptage)", bold_prefix="Coûteuse ")
    add_bullet(doc, "le comptage visuel d'un troupeau de 200 têtes a une marge d'erreur de 5–10 %", bold_prefix="Erreur-prone : ")
    add_bullet(doc, "impossible à réaliser en continu (24/7), sur de grandes surfaces, ou de nuit", bold_prefix="Limitée : ")
    add_bullet(doc, "l'identification individuelle par numéro d'oreille nécessite une proximité physique", bold_prefix="Subjective : ")

    add_heading_styled(doc, "2.2 Limites des solutions existantes", level=2)
    add_para(doc, "Les systèmes RFID et colliers connectés dominent le marché professionnel. Ils présentent des contraintes :")
    add_table(doc,
        ["Solution", "Coût par tête", "Limitation principale"],
        [
            ["RFID passif (boucle d'oreille)", "3–5 €", "Lecture manuelle ou portail obligatoire"],
            ["RFID actif + capteurs", "80–150 €", "Investissement infrastructure (antennes)"],
            ["Collier connecté IoT", "100–300 €", "Batterie limitée (2–3 ans), confort animal"],
            ["Caméras thermiques", "> 10 000 €", "Coût élevé, portée limitée"],
        ],
        col_widths=[2.2, 1.4, 2.8], first_col_bold=True)

    add_heading_styled(doc, "2.3 L'approche vision par ordinateur", level=2)
    add_para(doc,
        "La vision par ordinateur offre une alternative non-invasive, sans capteur sur "
        "l'animal et disponible 24/7. Cependant, identifier individuellement des bovins "
        "visuellement est un défi majeur car :")
    add_bullet(doc, "les bovins d'une même race se ressemblent énormément")
    add_bullet(doc, "les conditions d'éclairage varient (jour/nuit, ombres, intérieur/extérieur)")
    add_bullet(doc, "les animaux se déplacent et s'occlusionnent mutuellement")
    add_bullet(doc, "le suivi doit être maintenu sur de longues durées et entre vidéos distinctes")

    add_callout(doc, "Le problème central",
        "Ré-identifier un même bovin à travers des vidéos différentes — c'est précisément "
        "ce problème que Boeuf Tracker résout à l'aide d'embeddings DINOv2.",
        color_hex=NAVY_HEX)


def section_objectifs(doc):
    add_heading_styled(doc, "3. Objectifs du projet", level=1)

    add_heading_styled(doc, "3.1 Objectif principal", level=2)
    add_callout(doc, "Mission",
        "Concevoir un système temps-réel capable de détecter, identifier individuellement "
        "et classer par race les bovins dans un flux vidéo, sans capteur sur l'animal.",
        color_hex=GREEN_HEX)

    add_heading_styled(doc, "3.2 Objectifs spécifiques initialement fixés", level=2)
    objectifs_init = [
        "Détection et segmentation des bovins par image (YOLO + instance segmentation)",
        "Suivi intra-vidéo (tracking) pour attribuer un ID temporaire par frame",
        "Ré-identification cross-vidéo : reconnaître un bovin vu dans la vidéo A dans la vidéo B",
        "Classification de race automatique (10 races françaises principales)",
        "Interface utilisateur web avec visualisation temps-réel",
        "Performance ≥ 15 FPS sur matériel grand public",
    ]
    for i, obj in enumerate(objectifs_init, 1):
        add_bullet(doc, obj, bold_prefix=f"{i}. ")

    add_heading_styled(doc, "3.3 Objectifs additionnels réalisés (au-delà du cahier des charges)", level=2)
    objectifs_add = [
        "Classification de comportements (7 états via Numba JIT)",
        "Tableau de bord analytique : KPIs, graphiques, carte de chaleur",
        "Profils individuels par bovin avec historique croisé-vidéos",
        "Application desktop Tauri cross-platform (macOS / Windows / Linux)",
        "Noms propres persistants : chaque bovin reçoit un nom unique jamais réutilisé",
        "Optimisation Apple Silicon : MLX/Metal pour YOLOv26, FP16 pour DINOv2",
        "Hot-reload des paramètres sans redémarrage",
        "Auto-récupération en cas de crash de la source vidéo",
        "Documentation complète : doc technique, rapport, 6 diagrammes, 3 présentations",
        "Benchmark intégré YOLO11 vs YOLO26 vs DINOv2",
    ]
    for i, obj in enumerate(objectifs_add, 1):
        add_bullet(doc, obj, bold_prefix=f"{i}. ")


def section_etat_art(doc):
    add_heading_styled(doc, "4. État de l'art", level=1)

    add_heading_styled(doc, "4.1 Détection d'objets : évolution des architectures", level=2)
    add_table(doc,
        ["Génération", "Modèles", "Vitesse", "Caractéristique"],
        [
            ["Two-stage", "Faster R-CNN (2015)", "Lente", "Régions candidates puis classification"],
            ["Single-stage v1", "SSD, YOLOv1-3", "Rapide", "Détection dense en une passe"],
            ["Single-stage v2", "YOLOv4-8", "Très rapide", "Anchor-free, CSP, PANet"],
            ["Single-stage v3", "YOLOv9-11 (2024)", "Ultra rapide", "Programmable gradient information"],
            ["MLX-native", "YOLOv26 (2025)", "Extrême", "Optimisé Apple Silicon Metal"],
        ],
        col_widths=[1.5, 1.8, 1.2, 2.3], first_col_bold=True)

    add_callout(doc, "Choix retenu",
        "YOLOv11s-seg (PyTorch) en fallback universel, YOLOv26s-seg sur MLX en production "
        "sur macOS — ce dernier étant environ 2,6× plus rapide que YOLO11 sur PyTorch/MPS.",
        color_hex=GREEN_HEX)

    add_heading_styled(doc, "4.2 Ré-identification visuelle (Re-ID)", level=2)
    add_para(doc,
        "La ré-identification consiste à reconnaître une même entité à travers des caméras "
        "ou vidéos distinctes. Trois familles d'approches existent :")
    add_bullet(doc, "entraînement d'un réseau triplet/siamese sur un jeu annoté — nécessite un dataset de bovins inexistant publiquement.", bold_prefix="Méthodes supervisées : ")
    add_bullet(doc, "utilisation d'un modèle pré-entraîné générique et extraction de ses embeddings — approche choisie avec DINOv2-small (384-dim).", bold_prefix="Transfer learning : ")
    add_bullet(doc, "DINOv2 est auto-supervisé (DINO = self-DIstillation with NO labels), pré-entraîné sur 142 millions d'images — idéal sans données annotées.", bold_prefix="Self-supervised : ")

    add_para(doc,
        "Pourquoi DINOv2 plutôt que CLIP ? DINOv2 produit des features d'apparence pure "
        "(non alignées avec le langage), ce qui le rend supérieur pour la discrimination "
        "visuelle fine. CLIP aligne image et texte — utile pour la classification de race "
        "mais moins optimal pour la distinction d'individus proches.")

    add_heading_styled(doc, "4.3 Classification zero-shot de races", level=2)
    add_para(doc, "La classification zero-shot permet de reconnaître des classes sans aucune image d'entraînement. Évolution des modèles :")
    add_bullet(doc, "alignement image-texte sur 400M paires (ViT-B/32)", bold_prefix="CLIP (OpenAI, 2021) : ")
    add_bullet(doc, "sigmoid loss au lieu de softmax loss, plus scalable", bold_prefix="SigLIP (Google, 2023) : ")
    add_bullet(doc, "état de l'art actuel. La variante So400m (1136M paramètres) atteint 84,1 % top-1 sur ImageNet en zero-shot.", bold_prefix="SigLIP-2 (Google, 2025) : ")

    add_callout(doc, "Choix retenu",
        "SigLIP-2 So400m (google/siglip2-so400m-patch16-384) — le modèle le plus performant "
        "en 2025 pour la classification zero-shot fine-grained. Aucune image d'entraînement "
        "de race n'étant disponible, le zero-shot est la seule option viable.",
        color_hex=GREEN_HEX)


def section_architecture(doc):
    add_heading_styled(doc, "5. Architecture globale du système", level=1)

    add_heading_styled(doc, "5.1 Vue d'ensemble à deux processus", level=2)
    add_para(doc, "Le système suit une architecture client-serveur à deux processus :")
    add_image(doc, "diagram_architecture.png", width_inches=6.3,
              caption="Figure 2 — Architecture détaillée : wrapper Tauri supervise le worker Python sur le port 8100")

    add_para(doc,
        "Le worker Python est auto-suffisant : il sert à la fois l'API REST et l'interface "
        "web. Tauri n'est qu'un wrapper qui lance et supervise ce worker. Cette approche "
        "permet au worker de fonctionner indépendamment (accessible via navigateur, sans Tauri).")

    add_heading_styled(doc, "5.2 Choix de conception : worker HTTP plutôt qu'IPC", level=2)
    add_para(doc, "Contrairement à l'approche Tauri classique (IPC entre Rust et frontend via invoke), le projet communique entièrement via HTTP localhost. Avantages :")
    add_bullet(doc, "le worker fonctionne sans Tauri (accessible via navigateur)", bold_prefix="Indépendance : ")
    add_bullet(doc, "curl localhost:8100/api/stats à tout moment", bold_prefix="Debug facilité : ")
    add_bullet(doc, "pas de schéma IPC à maintenir", bold_prefix="Simplicité : ")
    add_bullet(doc, "entre mode web et mode desktop", bold_prefix="Frontend inchangé : ")

    add_heading_styled(doc, "5.3 Couches logicielles", level=2)
    add_table(doc,
        ["Couche", "Technologie", "Rôle"],
        [
            ["IA", "YOLOv26, DINOv2, SigLIP-2, Numba", "Inférence Machine Learning"],
            ["Logique", "Python (processor.py, detector.py...)", "Orchestration du pipeline"],
            ["Persistance", "Pickle (.pkl), JSON (counter, history)", "Stockage embeddings / données"],
            ["API", "Flask 3.x", "Serveur HTTP REST"],
            ["UI", "HTML5, CSS3, JavaScript vanilla", "Interface utilisateur"],
            ["Desktop", "Tauri v2 (Rust)", "Wrapper natif cross-platform"],
            ["Optionnelle", "Bun + Hono", "Proxy de développement"],
        ],
        col_widths=[1.3, 2.5, 2.7], first_col_bold=True)

    add_heading_styled(doc, "5.4 Diagramme du pipeline", level=2)
    add_image(doc, "diagram_pipeline.png", width_inches=6.3,
              caption="Figure 3 — Pipeline de traitement par frame (avec parties asynchrones)")


def section_detection(doc):
    add_heading_styled(doc, "6. Détection YOLO et segmentation d'instances", level=1)

    add_heading_styled(doc, "6.1 Rôle de la détection", level=2)
    add_para(doc, "La détection est la première étape du pipeline. Pour chaque frame vidéo, YOLO identifie les bovins en :")
    add_bullet(doc, "localisant leur boîte englobante (bounding box [x1, y1, x2, y2])")
    add_bullet(doc, "segmentant leur silhouette (mask au niveau pixel)")
    add_bullet(doc, "assignant un score de confiance (conf ∈ [0, 1])")
    add_bullet(doc, "maintenant un ID de suivi temporaire (track_id)")

    add_heading_styled(doc, "6.2 Backends de détection", level=2)

    add_heading_styled(doc, "Backend PyTorch (CattleDetector)", level=3)
    add_bullet(doc, "YOLOv11s-seg via Ultralytics, poids yolo11s-seg.pt (20,7 Mo)")
    add_bullet(doc, "Tracker ByteTrack (tracker='bytetrack.yaml')")
    add_bullet(doc, "Device auto : cuda > mps > cpu")
    add_bullet(doc, "FP16 activé sur cuda et mps (conversion manuelle pour compatibilité Ultralytics 8.4+)")
    add_bullet(doc, "Warmup sur image dummy 64×64 pour initialiser les kernels")

    add_heading_styled(doc, "Backend MLX (CattleDetectorMLX) — production macOS", level=3)
    add_bullet(doc, "YOLOv26s-seg via yolo26mlx, poids yolo26s-seg.safetensors (46,3 Mo)")
    add_bullet(doc, "Device : mlx (toujours Metal GPU)")
    add_bullet(doc, "FP16 natif (MLX est FP16 by design)")

    add_callout(doc, "Pourquoi un tracker custom en MLX ?",
        "La bibliothèque yolo26mlx expose predict() et track(), mais TrackerManager.update() "
        "drop les masks lors du tracking (bug documenté). Le projet implémente donc un "
        "_SimpleIoUTracker maison et utilise predict() + tracking manuel.",
        color_hex=RED.__str__().replace("0x", "")[:6] if False else "CC0000")

    add_heading_styled(doc, "6.3 Le tracker IoU maison (_SimpleIoUTracker)", level=2)
    add_para(doc, "Un tracker multi-objet minimaliste basé sur l'IoU (Intersection over Union). Paramètres :")
    add_table(doc,
        ["Paramètre", "Valeur", "Rôle"],
        [
            ["iou_threshold", "0.3", "IoU minimum pour associer une détection à un track"],
            ["max_lost", "30 frames", "Nombre de frames avant suppression d'un track perdu"],
        ],
        col_widths=[1.8, 1.4, 3.3], first_col_bold=True)

    add_heading_styled(doc, "6.4 Compatibilité backend-agnostic", level=2)
    add_para(doc,
        "Le code consommateur utilise des appels de style PyTorch "
        "(result.boxes.xyxy.cpu().numpy()). Pour fonctionner aussi avec les résultats "
        "MLX (arrays numpy), une couche de shims de compatibilité est implémentée :")
    add_table(doc,
        ["Shim", "Rôle"],
        [
            ["_NumpyWrapper", "Fait qu'un array numpy ait .cpu() retournant un _CpuView"],
            ["_CpuView", "Supporte .numpy(), .int(), .cpu() comme un tensor torch"],
            ["_MLXBoxes", "Reproduit .xyxy, .id, .conf, .cls"],
            ["_MLXMasks", "Reproduit .data"],
            ["_MLXResult", "Wrapper de résultat complet compatible Ultralytics"],
        ],
        col_widths=[1.8, 4.7], first_col_bold=True)

    add_heading_styled(doc, "6.5 Classe COCO utilisée", level=2)
    add_para(doc,
        "YOLO est entraîné sur COCO (80 classes). La classe vache est l'ID 19 "
        "(COW_CLASS_ID = 19). La détection filtre donc les résultats pour ne garder que "
        "cette classe :")
    add_code_block(doc, 'model.track(frame, classes=[19], tracker="bytetrack.yaml", persist=True)')
    add_para(doc, "Le paramètre persist=True permet au tracker de maintenir l'état entre les appels (IDs cohérents intra-vidéo).")


def section_tracking(doc):
    add_heading_styled(doc, "7. Suivi multi-objets (tracking)", level=1)

    add_heading_styled(doc, "7.1 Distinction tracking vs Re-ID", level=2)
    add_table(doc,
        ["Concept", "Portée", "Durée", "Mécanisme"],
        [
            ["Tracking (ByteTrack / IoU)", "Intra-vidéo", "Frame à frame", "Continuité spatiale (IoU)"],
            ["Re-ID (DINOv2)", "Cross-vidéo", "Permanent", "Empreinte visuelle (embedding)"],
        ],
        col_widths=[2.2, 1.4, 1.4, 1.5], first_col_bold=True)

    add_para(doc,
        "Le tracking est éphémère : si un bovin sort du champ puis revient, ByteTrack perd "
        "son ID et lui en attribue un nouveau. Le Re-ID, lui, reconnaît l'empreinte "
        "visuelle et restaure l'identité permanente (Boeuf_042).")

    add_heading_styled(doc, "7.2 Le double système d'identité", level=2)
    add_para(doc, "Chaque bovin détecté possède deux identités simultanées :")
    add_bullet(doc, "entier temporaire, peut changer d'une frame à l'autre", bold_prefix="track_id (YOLO) : ")
    add_bullet(doc, "clé permanente stockée en DB, jamais réutilisée", bold_prefix="Boeuf_NNN (Re-ID) : ")

    add_heading_styled(doc, "7.3 Anti-double-comptage sur boucle vidéo", level=2)
    add_para(doc,
        "Quand une vidéo fichier arrive à la fin et reboucle (rewind), tous les bovins "
        "réapparaissent avec de nouveaux track_id. Sans protection, le système les "
        "compterait comme nouveaux. Solution : détection de rewind via CAP_PROP_POS_FRAMES.")
    add_code_block(doc, '''current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
if current_pos < last_pos:  # régression = rewind
    loop_detected_at_frame = frame_count
    track_id_to_name.clear()  # forcera le re-Re-ID''')
    add_para(doc,
        "Pendant les 60 frames suivant un rewind (loop_grace_frames), le seuil de Re-ID "
        "est abaissé de 0.70 à 0.45 (loop_threshold) pour être plus permissif et récupérer "
        "les identités existantes.")


def section_reid(doc):
    add_heading_styled(doc, "8. Ré-identification (Re-ID) avec DINOv2", level=1)

    add_heading_styled(doc, "8.1 Principe", level=2)
    add_para(doc,
        "La ré-identification repose sur la notion d'embedding : une représentation "
        "vectorielle dense de l'apparence visuelle d'un bovin, telle que deux images du "
        "même animal produisent des vecteurs proches (cosinus ≈ 1) et deux animaux "
        "différents produisent des vecteurs éloignés.")

    add_heading_styled(doc, "8.2 Architecture de l'embedding composite (464 dimensions)", level=2)
    add_image(doc, "diagram_reid.png", width_inches=6.2,
              caption="Figure 4 — Embedding composite Re-ID : DINOv2 + HSV + LBP (464 dimensions)")
    add_para(doc, "Plutôt qu'utiliser uniquement DINOv2, le système construit un embedding composite de 464 dimensions combinant trois signaux complémentaires :")
    add_table(doc,
        ["Composante", "Dimensions", "Poids", "Capture"],
        [
            ["DINOv2-small", "384", "0.70", "Apparence sémantique globale (forme, posture)"],
            ["Histogramme HSV", "48", "0.20", "Distribution de couleur de robe"],
            ["LBP (Local Binary Pattern)", "32", "0.10", "Texture fine, robuste à l'éclairage"],
            ["TOTAL", "464", "1.00", "—"],
        ],
        col_widths=[2.0, 1.2, 0.9, 2.4], first_col_bold=True)

    add_callout(doc, "Conseil de tuning",
        "Pour des races à robe unie (Angus noir, Charolaise blanche), où la couleur seule "
        "discrimine fortement, ajuster dino_weight=0.3 et hsv_weight=0.5.",
        color_hex=GREEN_HEX)

    add_heading_styled(doc, "8.3 DINOv2-small", level=2)
    add_bullet(doc, "Modèle : facebook/dinov2-small (HuggingFace Transformers)")
    add_bullet(doc, "Dimensions : 384")
    add_bullet(doc, "Pré-entraînement : self-supervised, 142M images")
    add_bullet(doc, "Pooling : moyenne sur la dimension des tokens (last_hidden_state.mean(dim=1))")
    add_bullet(doc, "Normalisation : L2 avec epsilon 1e-8")

    add_heading_styled(doc, "8.4 Histogramme HSV (48-dim)", level=2)
    add_code_block(doc, '''def _hsv_hist(self, crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = []
    for ch in range(3):  # H, S, V séparément
        h = cv2.calcHist([hsv], [ch], None, [16],
                         [0, 180] if ch == 0 else [0, 256])
        hist.append(h.flatten())
    return np.concatenate(hist)  # 48-dim, normalisé par somme''')

    add_heading_styled(doc, "8.5 LBP (Local Binary Pattern, 32-dim)", level=2)
    add_para(doc, "Le LBP est un descripteur de texture classique :")
    add_bullet(doc, "Pour chaque pixel, comparer aux 8 voisins → code binaire 8-bit")
    add_bullet(doc, "Calculer l'histogramme sur une grille 4×4 avec 8 bins → 128 dim")
    add_bullet(doc, "Réduire à 32 dim en moyennant par groupes de 4")
    add_para(doc, "Le calcul est vectorisé via np.roll pour les 8 décalages, et l'image est pré-floutée avec cv2.GaussianBlur((3,3), 0).")

    add_heading_styled(doc, "8.6 Similarité pondérée par composante", level=2)
    add_para(doc, "Plutôt qu'un simple dot product sur les 464 dim concaténées, le système calcule la similarité par composante puis combine :")
    add_code_block(doc, '''def compare(self, embedding_a, embedding_b):
    # Découper en 3 slots
    a_dino, b_dino = embedding_a[:384],    embedding_b[:384]
    a_hsv,  b_hsv  = embedding_a[384:432], embedding_b[384:432]
    a_lbp,  b_lbp  = embedding_a[432:],    embedding_b[432:]

    # Cosinus par composante
    sim_dino = cosine(a_dino, b_dino)
    sim_hsv  = cosine(a_hsv,  b_hsv)
    sim_lbp  = cosine(a_lbp,  b_lbp)

    # Somme pondérée
    return (0.70 * sim_dino + 0.20 * sim_hsv + 0.10 * sim_lbp)''')

    add_heading_styled(doc, "8.7 Seuils de décision", level=2)
    add_table(doc,
        ["Seuil", "Valeur", "Utilisation"],
        [
            ["threshold", "0.70", "Re-ID normal (haute confiance)"],
            ["loop_threshold", "0.55", "Pendant 60 frames après rewind (permissif)"],
            ["Match minimal", "0.55", "En dessous → nouveau bovin"],
        ],
        col_widths=[1.8, 1.0, 3.7], first_col_bold=True)

    add_heading_styled(doc, "8.8 Mise à jour EMA (Exponential Moving Average)", level=2)
    add_para(doc, "L'embedding d'un bovin évolue dans le temps (angle de vue, éclairage). Pour garder une référence à jour sans dérive :")
    add_code_block(doc, '''def update(self, name, embedding, alpha=0.2):
    old = self.animals[name]["embedding"]
    new = (1.0 - alpha) * old + alpha * embedding
    new = new / (np.linalg.norm(new) + 1e-8)  # re-normaliser''')

    add_heading_styled(doc, "8.9 Optimisations DINOv2", level=2)
    add_table(doc,
        ["Optimisation", "Gain estimé", "Condition"],
        [
            ["torch.compile(mode='reduce-overhead')", "+20–30 %", "CUDA, PyTorch ≥ 2.0"],
            ["FP16 (model.half())", "+20 % single / +10 % batch", "MPS"],
            ["Batch processing (1 forward pour N crops)", "Dominant", "Toujours"],
            ["@torch.no_grad()", "Évite le calcul de gradient", "Toujours"],
        ],
        col_widths=[3.0, 2.0, 1.5], first_col_bold=True)


def section_breed(doc):
    add_heading_styled(doc, "9. Classification de race avec SigLIP-2", level=1)

    add_heading_styled(doc, "9.1 Le défi de la classification sans données d'entraînement", level=2)
    add_para(doc,
        "Le projet ne disposait pas d'images annotées de races bovines. Entraîner un "
        "classifieur classique était donc impossible. La solution retenue est le "
        "zero-shot classification : décrire chaque race en langage naturel et laisser le "
        "modèle VLM (Vision-Language Model) la reconnaître sans entraînement.")

    add_heading_styled(doc, "9.2 Évolution des approches", level=2)
    add_image(doc, "diagram_breed.png", width_inches=6.2,
              caption="Figure 5 — Classification de race zero-shot avec SigLIP-2 So400m")

    add_heading_styled(doc, "9.3 Les 10 races supportées", level=2)
    add_table(doc,
        ["Race", "Robe", "Origine", "Usage"],
        [
            ["Holstein", "Pie noire-blanc", "Pays-Bas", "Laitière (n°1 mondiale)"],
            ["Charolaise", "Blanche crème", "Bourgogne", "Bouchère"],
            ["Limousine", "Fauve", "Limousin", "Bouchère"],
            ["Salers", "Fauve rouge", "Auvergne", "Mixte"],
            ["Angus", "Noire", "Écosse", "Bouchère"],
            ["Normande", "Pie brune", "Normandie", "Mixte"],
            ["Blonde d'Aquitaine", "Froment", "Aquitaine", "Bouchère"],
            ["Montbéliarde", "Pie rouge", "Franche-Comté", "Laitière"],
            ["Hereford", "Rouge blanc-face", "Angleterre", "Bouchère"],
            ["Aubrac", "Fauve", "Aubrac", "Mixte"],
        ],
        col_widths=[1.8, 1.5, 1.5, 1.7], first_col_bold=True)

    add_heading_styled(doc, "9.4 Algorithme de classification", level=2)
    add_code_block(doc, '''def classify(self, crop_bgr):
    # 1. Embedder l'image
    emb = self._embed_image(crop_bgr)  # L2-normalisé

    # 2. Similarités cosinus avec les prompts pré-calculés
    sims = emb.cpu() @ self.text_emb.cpu().T  # (10,)

    # 3. Softmax avec température faible
    probs = softmax(sims / 0.01)  # τ=0.01 → distribution piquée

    # 4. Top-1 et marge
    top1_idx = argmax(probs)
    margin = probs[top1] - probs[top2]

    return {"race": BREEDS[top1_idx], "confidence": probs[top1], "margin": margin}''')

    add_heading_styled(doc, "9.5 Le seuil d'honnêteté", level=2)
    add_para(doc, "Si la marge (différence entre la première et deuxième race) est inférieure à 0.15, le système refuse de deviner et retourne « Indéterminée ».")
    add_callout(doc, "Philosophie",
        "Il vaut mieux dire « je ne sais pas » que donner une mauvaise réponse. Les races "
        "fauve (Charolaise, Limousine, Salers, Blonde d'Aquitaine) sont visuellement très "
        "proximes. SigLIP-2, même performant, peut hésiter entre elles.",
        color_hex=NAVY_HEX)

    add_heading_styled(doc, "9.6 Pré-calcul des embeddings texte", level=2)
    add_para(doc, "Les prompts des 10 races sont encodés une seule fois au démarrage. Ainsi, dans la boucle vidéo, seule l'embedding image est calculé.")

    add_heading_styled(doc, "9.7 Chaîne de fallback", level=2)
    add_code_block(doc, '''_BreedEngine._load():
    try:
        # 1. SigLIP-2 So400m (1136M params) — état de l'art
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384")
    except:
        try:
            # 2. CLIP ViT-B/32 (fallback)
            self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        except:
            # 3. HSV heuristique (dernier recours)
            return None''')

    add_heading_styled(doc, "9.8 Performance", level=2)
    add_bullet(doc, "Latence : ~22 ms par classification (MPS, M1 Pro)")
    add_bullet(doc, "Fréquence : uniquement sur nouveau bovin (première détection)")
    add_bullet(doc, "Impact sur FPS : négligeable")


def section_names(doc):
    add_heading_styled(doc, "10. Génération de noms et persistance d'identité", level=1)

    add_heading_styled(doc, "10.1 Le problème de l'identité stable", level=2)
    add_para(doc, "Un bovin détecté dans la vidéo A doit garder le même nom quand il réapparaît dans la vidéo B, même après :")
    add_bullet(doc, "redémarrage de l'application")
    add_bullet(doc, "réinitialisation de la base de données")
    add_bullet(doc, "changement de vidéo source")
    add_bullet(doc, "rebouclage de la vidéo")

    add_heading_styled(doc, "10.2 Architecture en 3 couches", level=2)

    add_heading_styled(doc, "Couche 1 : Re-ID embedding (identification visuelle)", level=3)
    add_para(doc, "Produit une empreinte visuelle stable (voir section 8).")

    add_heading_styled(doc, "Couche 2 : Base de données par vidéo (.pkl)", level=3)
    add_para(doc, "Chaque vidéo a sa propre base de données :")
    add_code_block(doc, '''def db_path_for_source(source) -> str:
    # Webcam 0 → "cattle_db_webcam0.pkl"
    # Fichier video.mp4 → "cattle_db_video.pkl"
    # Nom > 50 chars → hash MD5 → "cattle_db_video_<hash>.pkl"''')

    add_heading_styled(doc, "Couche 3 : Compteur global (names_counter.json)", level=3)
    add_para(doc, "Un simple compteur persistant qui ne fait que croître :")
    add_code_block(doc, '''class GlobalCounter:
    def next(self) -> int:
        self.value += 1
        self.save()  # persiste immédiatement
        return self.value''')

    add_heading_styled(doc, "10.3 La pool de noms (~120 noms)", level=2)
    add_para(doc, "Le système attribue des noms propres lisibles plutôt que des IDs techniques. La pool contient environ 120 noms équilibrés :")
    add_bullet(doc, "Marguerite, Aurelius, Charline, Daphne, Bastille, Beethoven, Ulysse, Sybille...", bold_prefix="Européens (50) : ")
    add_bullet(doc, "noms Wolof, Bambara, Mandinka, Swahili, Yoruba", bold_prefix="Africains (50) : ")
    add_bullet(doc, "noms de minéraux/phénomènes (débordement)", bold_prefix="Extras (20) : ")

    add_heading_styled(doc, "10.4 Mapping clé → nom", level=2)
    add_para(doc, "Le mapping se fait par numéro de clé (Boeuf_004 → NAME_POOL[3]), pas par index de tri. Ainsi, après un changement de vidéo qui vide la DB, les nouveaux Boeuf_005+ obtiennent des noms différents des Boeuf_001-004 de la vidéo précédente.")


def section_pipeline(doc):
    add_heading_styled(doc, "11. Pipeline de traitement vidéo", level=1)

    add_heading_styled(doc, "11.1 Vue d'ensemble du detection_loop", level=2)
    add_para(doc,
        "Le cœur du système est la fonction detection_loop(args) dans processor.py "
        "(environ 600 lignes). C'est une boucle infinie qui : capture, détecte, calcule "
        "les embeddings, matche, classifie les comportements, annote, encode en JPEG, "
        "met à jour l'état global.")

    add_heading_styled(doc, "11.2 Boucle principale (par frame)", level=2)
    add_code_block(doc, '''while True:  # boucle extérieure = récupération
    try:
        while True:  # boucle intérieure = pipeline normal
            # 1. Hot-reload des settings
            apply_desired_settings()

            # 2. Lecture frame
            ret, frame = cap.read()

            # 3. Détection de rewind
            if current_pos < last_pos:
                track_id_to_name.clear()

            # 4. Skip adaptatif
            if frame_count % (effective_skip + 1) != 0:
                continue

            # 5. Détection YOLO (~12 ms MLX)
            result = detector.detect(frame)

            # 6. Collecte crops + Re-ID async (~5 ms batch)
            reid_worker.submit_batch(crops_to_submit)
            embeddings = reid_worker.collect_ready()

            # 7. Match / EMA / Annotate
            for tid, emb in embeddings.items():
                name, sim = db.match(emb, threshold=0.70)
                if name is None:  # nouveau bovin
                    key = next_bovin_key()
                    breed = classify_breed(crop)
                    db.add(key, emb, breed=breed)

            # 8. Comportements (Numba ~0.1 ms)
            behaviors = analyze_behavior(boxes, track_ids)

            # 9. JPEG encode off-thread (~3 ms)
            STATE["frame_jpg"] = jpeg_bytes

    except Exception as e:
        emit_event("CRASH", traceback)
        time.sleep(2)
        cap = open_capture(source)  # auto-récupération''')

    add_heading_styled(doc, "11.3 Le ReIDWorker asynchrone", level=2)
    add_para(doc,
        "Un thread daemon séparé découple DINOv2 (MPS) de la boucle vidéo (YOLO MLX). "
        "DINOv2 tourne sur MPS (PyTorch) tandis que YOLOv26 tourne sur MLX (Metal). Les "
        "deux backends GPU fonctionnent en parallèle — le worker exploite cette concordance.")
    add_bullet(doc, "taille max 64, drops silencieux si plein", bold_prefix="Queue : ")
    add_bullet(doc, "{track_id: embedding}", bold_prefix="Résultats : ")
    add_bullet(doc, "replie sur synchrone en cas de crash", bold_prefix="Fallback : ")


def section_comportements(doc):
    add_heading_styled(doc, "12. Classification des comportements", level=1)

    add_heading_styled(doc, "12.1 Les 7 comportements reconnus", level=2)
    add_table(doc,
        ["Code", "Comportement", "Description"],
        [
            ["0", "couché", "Allongé, immobile longuement"],
            ["1", "pâture", "Broute, tête basse"],
            ["2", "boit", "À proximité d'un point d'eau"],
            ["3", "immobile", "Debout sans bouger"],
            ["4", "marche", "Déplacement lent"],
            ["5", "court", "Déplacement rapide"],
            ["6", "rué", "Sprint, panique"],
        ],
        col_widths=[0.8, 1.8, 3.9], first_col_bold=True)

    add_heading_styled(doc, "12.2 Features cinématiques", level=2)
    add_table(doc,
        ["Feature", "Description"],
        [
            ["speed (px/s)", "Vitesse de déplacement du centroïde"],
            ["aspect (ratio)", "box_width / box_height (un bovin couché a aspect > 1.7)"],
            ["rel_y", "Position verticale relative (0 = haut, 1 = bas)"],
        ],
        col_widths=[2.0, 4.5], first_col_bold=True)

    add_heading_styled(doc, "12.3 Arbre de décision Numba JIT", level=2)
    add_code_block(doc, '''@_njit(cache=True)
def _classify_behavior(speed, aspect, rel_y, immobile_dur):
    if aspect > 1.7 and speed < 5.0 and immobile_dur > 3.0:
        return 0  # couché
    if speed < 6.0 and aspect > 1.4:
        return 1  # pâture
    if speed < 4.0 and aspect > 1.3 and rel_y > 0.6:
        return 2  # boit
    if speed < 5.0:
        return 3  # immobile
    if speed < 25.0:
        return 4  # marche
    if speed < 80.0:
        return 5  # court
    return 6  # rué''')

    add_heading_styled(doc, "12.4 Lissage par vote majoritaire", level=2)
    add_para(doc, "Les comportements instantanés sont bruités. Le système maintient un historique roulant de 5 frames par track et applique un vote majoritaire.")


def section_analytics(doc):
    add_heading_styled(doc, "13. Analyse de données et tableau de bord", level=1)

    add_heading_styled(doc, "13.1 Le module analytics", level=2)
    add_para(doc, "Le module analytics.py collecte en arrière-plan des données pour alimenter le tableau de bord, via un thread daemon qui échantillonne STATE toutes les 2 secondes.")

    add_heading_styled(doc, "13.2 Données collectées", level=2)
    add_table(doc,
        ["Type", "Détail", "Persistance"],
        [
            ["FPS history", "Toutes les 2s, 600 points max", "web/data/history.json"],
            ["Race counts", "Comptage par race détectée", "history.json"],
            ["Activity counts", "Distribution des comportements", "history.json"],
            ["Timeline events", "NEW/MATCH/CRASH/INFO, 200 max", "history.json"],
            ["Spatial samples", "Position (x, y) normalisée + coat_type", "En mémoire"],
        ],
        col_widths=[1.8, 2.7, 2.0], first_col_bold=True)

    add_heading_styled(doc, "13.3 Endpoints API analytics", level=2)
    add_table(doc,
        ["Route", "Données retournées"],
        [
            ["/api/dashboard", "KPIs, fps_history, race_counts, activity_counts, timeline, breed_colors"],
            ["/api/heatmap", "Grille 20×20 avec compte par cellule + coat_type"],
            ["/api/profiles", "Profils individuels : breed, count, videos, activities_pct"],
        ],
        col_widths=[1.8, 4.7], first_col_bold=True)


def section_heatmap(doc):
    add_heading_styled(doc, "14. Carte de chaleur spatiale (heatmap)", level=1)

    add_heading_styled(doc, "14.1 Principe", level=2)
    add_image(doc, "diagram_heatmap.png", width_inches=6.2,
              caption="Figure 6 — Carte de chaleur spatiale (grille 20×20 avec rendu kernel-density)")
    add_para(doc, "La heatmap représente la densité spatiale des bovins dans le champ de la caméra. Elle permet d'identifier :")
    add_bullet(doc, "les zones de repos (concentration élevée)")
    add_bullet(doc, "les zones de passage (concentration moyenne)")
    add_bullet(doc, "les zones évitées (concentration nulle)")

    add_heading_styled(doc, "14.2 Grille 20×20", level=2)
    add_para(doc, "Le système discrétise l'image en une grille de 20×20 = 400 cellules. Chaque détection incrémente le compteur de sa cellule.")

    add_heading_styled(doc, "14.3 Rendu canvas côté frontend", level=2)
    add_para(doc, "Le frontend dessine la heatmap avec un effet kernel-density (gaussian radial gradients) et blending additif (globalCompositeOperation='lighter'). Palette 3 stops : vert → ambre → rouge brique.")


def section_api(doc):
    add_heading_styled(doc, "15. Serveur Flask et API REST", level=1)

    add_heading_styled(doc, "15.1 Architecture du serveur", level=2)
    add_para(doc, "Le serveur Flask (app.py, 758 lignes) est le pivot central : thread principal serveur Flask, thread daemon detection_loop, JSON provider NumpyJSONProvider (sérialise numpy).")

    add_heading_styled(doc, "15.2 Routes API complètes", level=2)
    add_table(doc,
        ["Route", "Méthode", "Description"],
        [
            ["/", "GET", "Sert index.html"],
            ["/video_feed", "GET", "Snapshot JPEG unique (polling)"],
            ["/api/stats", "GET", "État temps-réel : fps, source, animaux, événements"],
            ["/api/devices", "GET", "Liste des devices (mlx, cuda, mps, cpu)"],
            ["/api/device", "POST", "Change de device à l'exécution"],
            ["/api/upload-video", "POST", "Upload d'une vidéo (multipart)"],
            ["/api/source/webcam", "POST", "Bascule vers webcam"],
            ["/api/videos", "GET", "Liste des vidéos disponibles"],
            ["/api/source/file", "POST", "Bascule vers un fichier vidéo"],
            ["/api/diag", "GET", "Dump diagnostique de l'état"],
            ["/api/db/reset", "POST", "Purge la DB de la source courante"],
            ["/api/rematch", "POST", "Force le re-Re-ID de tous les tracks"],
            ["/api/restart", "POST", "Redémarre le serveur"],
            ["/api/settings", "GET", "Settings actuels + désirés + disponibles"],
            ["/api/settings", "POST", "Hot-reload des paramètres"],
            ["/api/breeds", "GET", "Liste des 10 races"],
            ["/api/breeds/<name>", "GET", "Détail d'une race"],
            ["/api/animals", "GET", "Tous les animaux en DB"],
            ["/api/dashboard", "GET", "Données du tableau de bord"],
            ["/api/heatmap", "GET", "Grille spatiale 20×20"],
            ["/api/profiles", "GET", "Profils individuels par bovin"],
            ["/api/bench", "GET", "Benchmark YOLO11 vs YOLO26 vs DINOv2"],
        ],
        col_widths=[1.9, 0.9, 3.7], first_col_bold=True)

    add_heading_styled(doc, "15.3 Le pattern desired/current pour le hot-reload", level=2)
    add_para(doc, "Les settings suivent un pattern à deux états : desired_* (ce que l'utilisateur veut) et *_current (ce qui est appliqué). Le frontend affiche un badge « pending » tant que desired != current.")

    add_heading_styled(doc, "15.4 Le endpoint /video_feed — JPEG polling", level=2)
    add_para(doc,
        "Plutôt qu'un flux MJPEG (multipart), le système retourne un JPEG unique par "
        "requête. Raison : Cloudflare et certains proxies tuent les connexions keep-alive "
        "> 100s (erreur 524). Le polling JPEG évite ce problème.")


def section_ui(doc):
    add_heading_styled(doc, "16. Interface utilisateur web", level=1)

    add_heading_styled(doc, "16.1 Stack frontend", level=2)
    add_bullet(doc, "HTML5 sémantique (index.html, 13 Ko)", bold_prefix="HTML : ")
    add_bullet(doc, "CSS3 avec variables custom, dark/light theme (styles.css, 18 Ko)", bold_prefix="CSS : ")
    add_bullet(doc, "JavaScript vanilla, pas de framework (app.js, 33 Ko)", bold_prefix="JS : ")
    add_bullet(doc, "Chart.js 4.4 (CDN) pour les graphiques", bold_prefix="Graphiques : ")
    add_bullet(doc, "Canvas natif pour la heatmap", bold_prefix="Heatmap : ")

    add_heading_styled(doc, "16.2 Layout principal", level=2)
    add_para(doc, "L'interface est organisée en 4 zones : topbar (logo, méta, thème), barre source (sélecteur vidéo, webcam, device), grille principale (flux vidéo + sidebar), et panneaux Dashboard/Stats coulissants.")

    add_heading_styled(doc, "16.3 Panneau Dashboard (slide-in)", level=2)
    add_bullet(doc, "bovins uniques / bovins visibles / FPS moyen / uptime", bold_prefix="4 KPIs : ")
    add_bullet(doc, "10 min d'historique", bold_prefix="Graphique FPS : ")
    add_bullet(doc, "avec légende custom", bold_prefix="Donut des races : ")
    add_bullet(doc, "triées par fréquence décroissante", bold_prefix="Barres des activités : ")
    add_bullet(doc, "canvas natif, kernel-density", bold_prefix="Heatmap spatiale : ")
    add_bullet(doc, "filtrable (ALL/NEW/MATCH/CRASH/INFO)", bold_prefix="Timeline : ")

    add_heading_styled(doc, "16.4 Polling cadencé", level=2)
    add_table(doc,
        ["Fonction", "Fréquence", "Endpoint"],
        [
            ["refreshStats()", "1000 ms", "/api/stats"],
            ["refreshDashboard()", "3000 ms (si ouvert)", "/api/dashboard + /api/heatmap"],
            ["refreshProfiles()", "5000 ms (si ouvert)", "/api/profiles"],
            ["Stream vidéo", "~40 ms (25 FPS)", "/video_feed"],
        ],
        col_widths=[2.0, 2.0, 2.5], first_col_bold=True)


def section_tauri(doc):
    add_heading_styled(doc, "17. Application desktop Tauri", level=1)

    add_heading_styled(doc, "17.1 Pourquoi Tauri ?", level=2)
    add_table(doc,
        ["Critère", "Electron", "Tauri"],
        [
            ["Taille binaire", "~150 Mo", "~1.8 Mo"],
            ["RAM", "~200 Mo", "~50 Mo"],
            ["Backend", "Node.js (bundled)", "Rust (système)"],
            ["WebView", "Bundled Chromium", "Native (WKWebView/WebView2)"],
            ["Démarrage", "~2 s", "~0.3 s"],
        ],
        col_widths=[1.8, 2.3, 2.4], first_col_bold=True)
    add_para(doc, "Tauri est environ 80× plus léger qu'Electron — crucial pour une application qui lance déjà un worker Python gourmand.")

    add_heading_styled(doc, "17.2 Architecture Tauri du projet", level=2)
    add_para(doc, "Le fichier src-tauri/src/main.rs (~266 lignes) implémente le cycle de vie complet : splash → spawn worker → health check → navigation → kill au shutdown.")

    add_heading_styled(doc, "17.3 Health check en 2 phases", level=2)
    add_bullet(doc, "poll TCP connect sur 127.0.0.1:8100 jusqu'à ouverture", bold_prefix="Phase 1 : ")
    add_bullet(doc, "curl /api/stats et parser fps, attendre fps > 0 (modèles chargés)", bold_prefix="Phase 2 : ")
    add_callout(doc, "Pourquoi 2 phases ?",
        "Le port TCP s'ouvre dès que Flask démarre (~1s), mais les modèles MLX/SigLIP-2 "
        "prennent 15-30s à charger. Sans la phase 2, l'UI afficherait fps:0 pendant ce temps.",
        color_hex=NAVY_HEX)

    add_heading_styled(doc, "17.4 Splash screen", level=2)
    add_para(doc, "Pendant le démarrage du worker, Tauri affiche splash.html : logo BT pulsant, messages rotatifs toutes les 2.5s.")

    add_heading_styled(doc, "17.5 Configuration bundle", level=2)
    add_para(doc, "Le bundler Tauri génère des artefacts natifs pour chaque plateforme : macOS (.app/.dmg), Windows (.exe/.msi), Linux (.AppImage/.deb). Aucune cross-compilation — il faut builder sur chaque OS.")

    add_heading_styled(doc, "17.6 Limitation assumée : Python requis", level=2)
    add_para(doc, "Le bundle Tauri n'embarque pas Python ni les modèles. Il suppose Python 3.10+, un .venv configuré, et les poids des modèles. C'est un compromis assumé pour éviter un exécutable de plusieurs Go instable.")


def section_optimisations(doc):
    add_heading_styled(doc, "18. Optimisations de performance", level=1)

    add_heading_styled(doc, "18.1 Tableau récapitulatif", level=2)
    add_image(doc, "diagram_perf.png", width_inches=6.2,
              caption="Figure 7 — Optimisations de performance et budget temps par frame")
    add_table(doc,
        ["Optimisation", "Localisation", "Gain estimé"],
        [
            ["MLX/Metal pour YOLOv26", "detector.py", "2,6× vs PyTorch/MPS"],
            ["DINOv2 batch processing", "reid.py", "N forwards → 1 (dominant)"],
            ["ReIDWorker asynchrone", "reid_worker.py", "Découple Re-ID du loop"],
            ["torch.compile (CUDA)", "reid.py", "+20–30 %"],
            ["FP16 sur MPS", "reid.py", "+20 %"],
            ["Numba JIT comportements", "processor.py", "×5–10 vs Python pur"],
            ["Annotation ROI-scoped", "processor.py", "29 ms → 9,5 ms/frame"],
            ["BLAS vectorisé match()", "database.py", "O(N) → 1 BLAS"],
            ["Skip adaptatif", "processor.py", "25 FPS garanti"],
            ["JPEG encode off-thread", "processor.py", "−3 ms/frame"],
            ["EMA cap 30 updates", "processor.py", "Borne la dérive"],
            ["Pré-calcul embeddings texte", "breed.py", "0 ms in-loop"],
        ],
        col_widths=[2.5, 1.6, 2.4], first_col_bold=True)

    add_heading_styled(doc, "18.2 Détail : annotation ROI-scoped", level=2)
    add_para(doc, "L'annotation naïve dessine le mask sur toute l'image (~29 ms). L'optimisation restreint l'opération à la bounding box uniquement (~9,5 ms). Gain : −65 % sur l'annotation.")

    add_heading_styled(doc, "18.3 Détail : match() vectorisé", level=2)
    add_para(doc, "Version naïve : N dot products Python. Version vectorisée : 1 opération matricielle BLAS. Pour N = 50 bovins, la vectorisation est environ 10× plus rapide.")

    add_heading_styled(doc, "18.4 Détail : skip adaptatif", level=2)
    add_para(doc, "Si fps < 20 pendant 3 frames consécutives, _current_skip augmente (max 4). Si fps > 28 pendant 5 frames, il diminue. Objectif : maintenir 25 FPS en sacrifiant des frames si nécessaire.")


def section_robustesse(doc):
    add_heading_styled(doc, "19. Robustesse et gestion d'erreurs", level=1)

    add_heading_styled(doc, "19.1 Auto-récupération du pipeline", level=2)
    add_para(doc, "La detection_loop est enveloppée dans deux boucles while : la boucle intérieure exécute le pipeline normal, la boucle extérieure attrape les exceptions, logge un event CRASH, attend 2s et reprend. Un crash ne tue jamais le système.")

    add_heading_styled(doc, "19.2 Chaîne de fallback des modèles", level=2)
    add_para(doc, "Chaque transition est try/except guarded :")
    add_bullet(doc, "YOLOv26 MLX → YOLOv11 PyTorch (si MLX indisponible)", bold_prefix="Détection : ")
    add_bullet(doc, "DINOv2 + HSV + LBP → DINOv2 seul → HSV seul", bold_prefix="Re-ID : ")
    add_bullet(doc, "SigLIP-2 So400m → CLIP ViT-B/32 → HSV heuristique", bold_prefix="Race : ")
    add_bullet(doc, "MLX (Metal) → CUDA → MPS → CPU", bold_prefix="Device : ")

    add_heading_styled(doc, "19.3 Récupération de source vidéo", level=2)
    add_para(doc, "Webcam : reconnect après 2s + warmup 3s. Fichier : rewind à frame 0.")

    add_heading_styled(doc, "19.4 Validation des dimensions d'embedding", level=2)
    add_para(doc, "Si l'utilisateur change de modèle Re-ID (changement de dimension), la DB détecte et purge automatiquement les embeddings incompatibles.")


def section_evolution(doc):
    add_heading_styled(doc, "20. Évolution chronologique du projet", level=1)

    add_heading_styled(doc, "20.1 Timeline des 55 commits", level=2)
    add_table(doc,
        ["Date", "Commits", "Thématique"],
        [
            ["24 juin 2026", "4", "MVP initial (Flask + YOLO + DINOv2 + UI DaisyUI)"],
            ["26 juin 2026", "3", "Settings live + notebook Colab"],
            ["27 juin 2026", "18", "Bataille Cloudflare/streaming (JPEG polling, Gradio, ngrok)"],
            ["13 juillet 2026", "16", "Checkpoints intermédiaires"],
            ["15 juillet 2026", "13", "Vague de features majeures (perf, UI, noms, breeds, Tauri)"],
            ["16 juillet 2026", "1", "Documentation finale + PPTX"],
        ],
        col_widths=[1.8, 1.0, 3.7], first_col_bold=True)

    add_heading_styled(doc, "20.2 Évolution des modèles de classification de race", level=2)
    add_para(doc, "Chaque itération a résolu une limitation de la précédente :")
    add_bullet(doc, "trop imprécis (Charolaise = Limousine par couleur)", bold_prefix="HSV heuristique → ")
    add_bullet(doc, "sigmoid plat, marges toujours < 0.04 → tout « Indéterminée »", bold_prefix="CLIP ViT-B/32 → ")
    add_bullet(docétat_de_l_art := "softmax τ=0.01, marges exploitables — version finale", bold_prefix="SigLIP-2 So400m + softmax → ") if False else add_bullet(doc, "softmax τ=0.01, marges exploitables — version finale", bold_prefix="SigLIP-2 So400m + softmax → ")


def section_resultats(doc):
    add_heading_styled(doc, "21. Résultats et métriques", level=1)

    add_heading_styled(doc, "21.1 Performance temps-réel", level=2)
    add_table(doc,
        ["Métrique", "Valeur", "Objectif", "Statut"],
        [
            ["FPS moyen (M1 Pro)", "23–26 FPS", "15 FPS", "Dépassé"],
            ["Latence détection YOLO", "~12 ms", "< 30 ms", "Atteint"],
            ["Latence Re-ID (batch)", "~5 ms async", "< 10 ms", "Atteint"],
            ["Latence breed (SigLIP-2)", "~22 ms", "< 50 ms", "Atteint"],
            ["Latence comportement", "~0.1 ms", "< 1 ms", "Atteint"],
            ["Latence annotation ROI", "~9,5 ms", "< 15 ms", "Atteint"],
            ["Total par frame", "~38 ms", "< 67 ms", "Atteint"],
        ],
        col_widths=[2.2, 1.5, 1.3, 1.2], first_col_bold=True)

    add_heading_styled(doc, "21.2 Métriques de classification de race", level=2)
    add_para(doc, "Cas type (Normande vs Angus) — comparaison sigmoid vs softmax :")
    add_table(doc,
        ["Approche", "Normande", "Angus", "Marge", "Verdict"],
        [
            ["Sigmoid (avant)", "0.51", "0.49", "0.02", "Échec"],
            ["Softmax τ=0.01 (après)", "0.63", "0.19", "0.44", "Succès"],
        ],
        col_widths=[2.2, 1.2, 1.0, 1.0, 1.1], first_col_bold=True)

    add_heading_styled(doc, "21.3 Empreinte ressources", level=2)
    add_table(doc,
        ["Ressource", "Valeur"],
        [
            ["Binaire Tauri", "~1.8 Mo"],
            ["RAM worker Python", "~2–3 Go (modèles chargés)"],
            ["Poids YOLOv26 MLX", "46 Mo"],
            ["Poids DINOv2-small", "~85 Mo"],
            ["Poids SigLIP-2 So400m", "~4.5 Go"],
            ["DB par vidéo", "~10–20 Ko"],
        ],
        col_widths=[2.5, 3.5], first_col_bold=True)

    add_heading_styled(doc, "21.4 Lignes de code", level=2)
    add_table(doc,
        ["Composant", "Lignes"],
        [
            ["Python (core)", "~4 700"],
            ["JavaScript (frontend)", "~750"],
            ["Rust (Tauri)", "~266"],
            ["CSS", "~700"],
            ["HTML", "~400"],
            ["Documentation markdown", "~1 200"],
            ["Total", "~8 000"],
        ],
        col_widths=[3.0, 2.0], first_col_bold=True)

    add_heading_styled(doc, "21.5 Couverture fonctionnelle", level=2)
    add_para(doc, "Bilan : 6/6 objectifs initiaux + 10/10 objectifs additionnels = 16/16.")


def section_limites(doc):
    add_heading_styled(doc, "22. Limites connues", level=1)

    add_heading_styled(doc, "22.1 Classification de race", level=2)
    add_bullet(doc, "Charolaise/Limousine/Salers/Blonde d'Aquitaine ont des robes similaires → ~30 % classés « Indéterminée »", bold_prefix="Confusion races fauve : ")
    add_bullet(doc, "le zero-shot est limité par la qualité des prompts", bold_prefix="Pas de fine-tuning : ")

    add_heading_styled(doc, "22.2 Re-ID", level=2)
    add_bullet(doc, "dans un troupeau de Charolaises toutes blanches, DINOv2 peut confondre deux individus", bold_prefix="Bovins très similaires : ")
    add_bullet(doc, "peut rater des matches si l'angle de vue diffère fortement", bold_prefix="Seuil 0.70 strict : ")

    add_heading_styled(doc, "22.3 Déploiement", level=2)
    add_bullet(doc, "le bundle Tauri n'embarque pas Python", bold_prefix="Python requis : ")
    add_bullet(doc, "SigLIP-2 (~4.5 Go) au premier lancement", bold_prefix="Modèles volumineux : ")
    add_bullet(doc, "sur Windows/Linux, fallback PyTorch/CUDA moins performant", bold_prefix="MLX = macOS : ")


def section_perspectives(doc):
    add_heading_styled(doc, "23. Perspectives et améliorations futures", level=1)

    add_heading_styled(doc, "23.1 Améliorations IA", level=2)
    add_bullet(doc, "collecter un dataset annoté de races françaises et fine-tuner SigLIP-2 ou EfficientNet", bold_prefix="Fine-tuning race : ")
    add_bullet(doc, "ré-entraîner DINOv2 sur un dataset de bovins (transfer learning)", bold_prefix="Amélioration Re-ID : ")
    add_bullet(doc, "body condition scoring, détection de boiterie, détection de chaleur", bold_prefix="Détection d'health : ")

    add_heading_styled(doc, "23.2 Améliorations système", level=2)
    add_bullet(doc, "synchroniser plusieurs sources + cross-camera Re-ID", bold_prefix="Multi-caméras : ")
    add_bullet(doc, "Jetson Nano/Orin (TensorRT), Raspberry Pi 5 (quantification INT8)", bold_prefix="Edge deployment : ")
    add_bullet(doc, "notification si bovin absent, alerte comportement anormal", bold_prefix="Real-time alerts : ")

    add_heading_styled(doc, "23.3 Améliorations UX", level=2)
    add_bullet(doc, "application compagnon iOS/Android pour consultation à distance", bold_prefix="Mobile app : ")
    add_bullet(doc, "export PDF quotidien/hebdomadaire, statistiques de cheptel", bold_prefix="Rapports automatisés : ")


def section_installation(doc):
    add_heading_styled(doc, "24. Guide d'installation et de déploiement", level=1)

    add_heading_styled(doc, "24.1 Prérequis", level=2)
    add_bullet(doc, "macOS 13.0+, Python 3.10+, Apple Silicon (M1/M2/M3) pour MLX", bold_prefix="macOS : ")
    add_bullet(doc, "Windows 10/11, Python 3.10+, GPU NVIDIA recommandé (CUDA)", bold_prefix="Windows : ")
    add_bullet(doc, "Ubuntu 22.04+, Python 3.10+, GPU NVIDIA recommandé", bold_prefix="Linux : ")

    add_heading_styled(doc, "24.2 Installation", level=2)
    add_code_block(doc, '''# 1. Cloner le dépôt
git clone <repo-url> boeuf-tracker
cd boeuf-tracker

# 2. Créer le venv
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\\Scripts\\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (macOS) Installer MLX
pip install mlx mlx-vlm

# 5. Lancer le worker
python app.py --mlx --port 8100

# 6. Accéder à l'UI
open http://localhost:8100''', language="bash")

    add_heading_styled(doc, "24.3 Build de l'app desktop Tauri", level=2)
    add_code_block(doc, '''# Prérequis : Rust + Tauri CLI
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2.0"

# Build
./build.sh
# Output : src-tauri/target/release/bundle/
#   macOS :  Boeuf Tracker.app + .dmg
#   Win :    Boeuf Tracker.exe + .msi
#   Linux :  boeuf-tracker.AppImage + .deb''', language="bash")

    add_heading_styled(doc, "24.4 Variables CLI importantes", level=2)
    add_table(doc,
        ["Flag", "Défaut", "Description"],
        [
            ["--source", "auto", "Chemin vidéo ou index webcam"],
            ["--port", "8100", "Port HTTP"],
            ["--yolo-model", "auto", "yolo26s-seg.safetensors si présent"],
            ["--threshold", "0.70", "Seuil Re-ID normal"],
            ["--loop-threshold", "0.55", "Seuil Re-ID post-rewind"],
            ["--conf", "0.4", "Confiance YOLO minimum"],
            ["--device", "auto", "mlx/cuda/mps/cpu"],
            ["--imgsz", "640", "Résolution YOLO"],
            ["--embed-every", "10", "Frames entre re-embeds"],
        ],
        col_widths=[1.8, 1.0, 3.7], first_col_bold=True)


def section_structure(doc):
    add_heading_styled(doc, "25. Structure du code source", level=1)
    add_para(doc, "Le projet suit une structure modulaire avec une dualité legacy/OOP :")
    add_bullet(doc, "app.py, processor.py, detector.py, reid.py, breed.py, names.py... — chemin d'exécution actuel", bold_prefix="Modules root : ")
    add_bullet(doc, "core/, utils/, models/, api/ — refactor orienté objet en cours", bold_prefix="Packages OOP : ")
    add_bullet(doc, "web/public/ — HTML/JS/CSS vanilla actifs", bold_prefix="Frontend vanilla : ")
    add_bullet(doc, "src-tauri/ — wrapper Rust/Tauri", bold_prefix="Desktop : ")


def section_glossaire(doc):
    add_heading_styled(doc, "26. Glossaire technique", level=1)
    add_table(doc,
        ["Terme", "Définition"],
        [
            ["Bounding box", "Rectangle délimitant un objet détecté [x1, y1, x2, y2]"],
            ["Segmentation d'instance", "Attribution pixel-par-pixel de chaque objet"],
            ["IoU", "Intersection over Union = intersection / union de deux boxes"],
            ["Tracker", "Algorithme maintenant un ID entre frames (ByteTrack, IoU)"],
            ["Embedding", "Représentation vectorielle dense d'une image (ex. 464-dim)"],
            ["Cosine similarity", "dot(a,b)/(norm(a)*norm(b)), ∈ [-1, 1]"],
            ["Re-ID", "Ré-identification : reconnaître le même individu à travers vidéos"],
            ["Zero-shot", "Classification sans images d'entraînement de la classe"],
            ["VLM", "Vision-Language Model (CLIP, SigLIP)"],
            ["Softmax", "Fonction convertissant un vecteur en distribution de probabilité"],
            ["Temperature", "Paramètre τ contrôlant la « sharpness » du softmax"],
            ["Margin", "Différence top1 − top2 dans une distribution"],
            ["DINOv2", "Modèle self-supervised de Meta (2023) produisant des embeddings"],
            ["SigLIP-2", "VLM de Google (2025), état de l'art zero-shot"],
            ["MLX", "Framework ML d'Apple pour Silicon (Metal GPU natif)"],
            ["MPS", "Metal Performance Shaders (backend PyTorch sur Metal)"],
            ["FP16", "Flottant demi-précision (16-bit), 2× plus rapide que FP32"],
            ["EMA", "Exponential Moving Average : new = (1-α)*old + α*input"],
            ["BLAS", "Basic Linear Algebra Subprograms (opérations matricielles)"],
            ["Numba JIT", "Just-In-Time compiler Python → LLVM"],
            ["LBP", "Local Binary Pattern (descripteur de texture)"],
            ["HSV", "Hue Saturation Value (espace couleur)"],
            ["ByteTrack", "Tracker multi-objets basé IoU (Ultralytics)"],
            ["Sidecar", "Processus enfant lancé et supervisé par un parent"],
            ["Hot-reload", "Modification de paramètres sans redémarrage"],
        ],
        col_widths=[1.8, 4.7], first_col_bold=True)


def section_references(doc):
    add_heading_styled(doc, "27. Références", level=1)

    add_heading_styled(doc, "Modèles et papers", level=2)
    refs = [
        "YOLOv11 — Ultralytics (2024). Documentation officielle.",
        "DINOv2 — Oquab et al. (Meta AI, 2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193",
        "SigLIP — Zhai et al. (Google, 2023). Sigmoid Loss for Language Image Pre-Training. arXiv:2303.15343",
        "SigLIP-2 — Tschannen et al. (Google, 2025). SigLIP 2: Multilingual Vision-Language Encoders. arXiv:2502.14786",
        "ByteTrack — Zhang et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. arXiv:2110.06864",
        "CLIP — Radford et al. (OpenAI, 2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020",
        "COCO Dataset — Lin et al. (2014). Microsoft COCO: Common Objects in Context.",
    ]
    for r in refs:
        add_bullet(doc, r)

    add_heading_styled(doc, "Technologies", level=2)
    techs = [
        "MLX — Apple. MLX: An Array Framework for Machine Learning on Apple Silicon.",
        "PyTorch — Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.",
        "Transformers — Wolf et al. (2020). HuggingFace Transformers.",
        "Flask — Pallets Projects.",
        "Tauri — Tauri Programme. Tauri v2 Documentation.",
        "Numba — Lam et al. (2015). Numba: A LLVM-based Python JIT Compiler.",
        "Chart.js — Simple yet flexible JavaScript charting.",
        "Bun — Oven-sh. Incredibly fast JavaScript runtime.",
        "Hono — Ultrafast Web Framework for the Edges.",
    ]
    for t in techs:
        add_bullet(doc, t)


def section_conclusion(doc):
    add_heading_styled(doc, "Conclusion", level=1)
    add_para(doc,
        "Le projet Boeuf Tracker démontre la viabilité d'une approche vision par ordinateur "
        "pour la surveillance d'élevage bovin, en combinant quatre technologies d'IA état "
        "de l'art (YOLOv26 MLX, DINOv2, SigLIP-2, Numba) dans un système cohérent et "
        "performant.")

    add_para(doc,
        "Les résultats dépassent les objectifs initiaux : 23–26 FPS (vs 15 visés), "
        "16 fonctionnalités livrées (vs 6 prévues), application desktop cross-platform, "
        "dashboard analytique complet, et classification de race zero-shot sans données "
        "d'entraînement.")

    add_para(doc,
        "La philosophie d'honnêteté (seuil de marge 0.15 pour refuser de deviner une race "
        "incertaine) et la persistance d'identité (compteur global jamais réutilisé, DB "
        "par vidéo) témoignent d'une conception prudente et robuste.")

    add_para(doc,
        "Les perspectives (fine-tuning race, multi-caméras, edge deployment, capteurs "
        "additionnels) ouvrent la voie à un produit commercialisable, tout en restant "
        "fondées sur une architecture éprouvée et documentée. Ce projet illustre "
        "l'intégration réussie de technologies IA modernes dans un cas d'usage concret "
        "et utile, avec une attention particulière à la performance, la robustesse et "
        "l'expérience utilisateur — des compétences centrales du génie électrique contemporain.")

    # Pied de page final
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(30)
    r = p.add_run("________________________________________")
    style_run(r, size=12, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Fin du rapport")
    style_run(r, size=12, bold=True, color=GREY)


# ============================================================================
# EN-TÊTE / PIED DE PAGE
# ============================================================================

def add_headers_footers(doc):
    section = doc.sections[0]

    # En-tête
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("Boeuf Tracker — Rapport Technique Complet")
    style_run(r, size=8.5, color=GREY, italic=True)

    # Pied de page avec numéro de page
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run("Page ")
    style_run(r1, size=9, color=GREY)
    # Champ PAGE
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    r2 = p.add_run()
    r2._r.append(fldChar1)
    r2._r.append(instrText)
    r2._r.append(fldChar2)
    style_run(r2, size=9, color=GREY)


# ============================================================================
# MAIN
# ============================================================================

def main():
    doc = Document()
    configure_document(doc)

    # Couverture
    build_cover(doc)

    # Table des matières
    build_toc(doc)

    # Sections
    section_resume(doc)
    add_page_break(doc)
    section_contexte(doc)
    add_page_break(doc)
    section_objectifs(doc)
    add_page_break(doc)
    section_etat_art(doc)
    add_page_break(doc)
    section_architecture(doc)
    add_page_break(doc)
    section_detection(doc)
    add_page_break(doc)
    section_tracking(doc)
    add_page_break(doc)
    section_reid(doc)
    add_page_break(doc)
    section_breed(doc)
    add_page_break(doc)
    section_names(doc)
    add_page_break(doc)
    section_pipeline(doc)
    add_page_break(doc)
    section_comportements(doc)
    add_page_break(doc)
    section_analytics(doc)
    add_page_break(doc)
    section_heatmap(doc)
    add_page_break(doc)
    section_api(doc)
    add_page_break(doc)
    section_ui(doc)
    add_page_break(doc)
    section_tauri(doc)
    add_page_break(doc)
    section_optimisations(doc)
    add_page_break(doc)
    section_robustesse(doc)
    add_page_break(doc)
    section_evolution(doc)
    add_page_break(doc)
    section_resultats(doc)
    add_page_break(doc)
    section_limites(doc)
    add_page_break(doc)
    section_perspectives(doc)
    add_page_break(doc)
    section_installation(doc)
    add_page_break(doc)
    section_structure(doc)
    add_page_break(doc)
    section_glossaire(doc)
    add_page_break(doc)
    section_references(doc)
    add_page_break(doc)
    section_conclusion(doc)

    # En-tête / pied de page
    add_headers_footers(doc)

    doc.save(OUT)
    print(f"✅ Document généré : {OUT}")
    size = os.path.getsize(OUT) / 1024
    print(f"   Taille : {size:.0f} Ko")


if __name__ == "__main__":
    main()
