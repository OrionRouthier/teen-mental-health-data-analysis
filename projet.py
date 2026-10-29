import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ==========================================
print("Chargement des donnees...")
df = pd.read_csv('./Teen_Mental_Health_Dataset.csv', sep=',')
df.columns = df.columns.str.strip()

# Suppression des variables démographiques non comportementales pour éviter les biais
mots_cles_a_supprimer = ['gender', 'age', 'platform', 'academic']
cols_a_supprimer = [col for col in df.columns if any(kw in col.lower() for kw in mots_cles_a_supprimer)]
df = df.drop(columns=cols_a_supprimer, errors='ignore')
print(f"--> Variables supprimees : {cols_a_supprimer}\n")

# ==========================================
# 2. FEATURE ENGINEERING (CREATION DE VARIABLES)
# ==========================================
df['ratio_ecran_sommeil'] = df['daily_social_media_hours'] / df['sleep_hours']
df['charge_mentale_index'] = df['stress_level'] + df['anxiety_level'] + df['addiction_level']

# Indicateur de synergie critique : effet multiplicatif si le stress et l'anxiété dépassent le seuil critique (> 6)
produit_symptomes = df['stress_level'] * df['anxiety_level']
seuils_respectes = (df['stress_level'] > 6) & (df['anxiety_level'] > 6)
df['seuil_synergie_critique'] = np.where(seuils_respectes, produit_symptomes, 0)

df['etat_depressif_str'] = df['depression_label'].map({0: 'No Depression', 1: 'Depression'})

# Encodage des variables qualitatives pour l'analyse de corrélation globale
colonnes_dummies = [col for col in ['social_interaction_level', 'gender', 'platform_usage'] if col in df.columns]
df_num = pd.get_dummies(df, columns=colonnes_dummies, dtype=int)

sns.set_theme(style="whitegrid", palette="pastel")

# ==========================================
# 3. ANALYSE EXPLORATOIRE ET VISUALISATIONS
# ==========================================

# A. Vue globale 5D
g2 = sns.relplot(
    data=df, x="daily_social_media_hours", y="sleep_hours", size="stress_level",          
    hue="anxiety_level", col="depression_label", palette="viridis",            
    sizes=(20, 250), alpha=0.7, height=6, aspect=1.1                    
)
g2.set_axis_labels("Heures de Reseaux Sociaux / jour", "Heures de Sommeil / nuit")
g2.set_titles("Diagnostic Depression : {col_name}")
g2.fig.suptitle("Analyse 5D : Population globale", y=1.05, fontsize=14)
plt.show()

# Isolation du sous-ensemble "Zone de Danger" (Sommeil < 6h et Réseaux > 5h)
df_ajuste = df[(df['sleep_hours'] < 6) & (df['daily_social_media_hours'] > 5)].copy()
print(f"Nombre d'adolescents dans la zone de danger : {len(df_ajuste)}")

# B. Corrélations globales avec les nouvelles variables
plt.figure(figsize=(10, 6))
df_pour_calcul = df_num.drop(columns=['etat_depressif_str'], errors='ignore')
corr_avec_depression = df_pour_calcul.corr()[['depression_label']].drop('depression_label').sort_values(by='depression_label', ascending=False)

sns.barplot(x=corr_avec_depression['depression_label'], y=corr_avec_depression.index, palette="vlag")
plt.title("Matrice de correlation (Incluant les features artisanales)", fontsize=14, pad=15)
plt.xlabel("Force de la correlation (Positif = Aggravant, Negatif = Protecteur)")
plt.ylabel("Variables etudiees")
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# C. Distributions par Boîtes à Moustaches (Boxplots)
compare_cols = ['ratio_ecran_sommeil', 'charge_mentale_index', 'seuil_synergie_critique',
                'daily_social_media_hours', 'sleep_hours', 'stress_level']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
ordre_affichage = ['No Depression', 'Depression']

for ax, col in zip(axes, compare_cols):
    sns.boxplot(
        data=df_ajuste, x='etat_depressif_str', y=col,
        palette={'No Depression': '#2ecc71', 'Depression': '#e74c3c'},
        order=ordre_affichage, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 7},
        ax=ax, width=0.5
    )
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('')

plt.suptitle("Distributions statistiques selon l'etat depressif (Box Plots — ◆ = Moyenne)", fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# D. Cartographie 3D des Handcrafted Features
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

sains = df[df['depression_label'] == 0]
depressifs = df[df['depression_label'] == 1]

ax.scatter(sains['ratio_ecran_sommeil'], sains['seuil_synergie_critique'], sains['charge_mentale_index'], c='#2ecc71', label='Sain (0)', s=40, alpha=0.5, edgecolors='black', linewidth=0.4)
ax.scatter(depressifs['ratio_ecran_sommeil'], depressifs['seuil_synergie_critique'], depressifs['charge_mentale_index'], c='#e74c3c', label='Depressif (1)', s=70, alpha=0.9, edgecolors='black', linewidth=0.7)

ax.set_xlabel("X : Ratio Ecran / Sommeil", fontsize=10, labelpad=12)
ax.set_ylabel("Y : Seuil Synergie Critique", fontsize=10, labelpad=12)
ax.set_zlabel("Z : Charge Mentale Index", fontsize=10, labelpad=12)
plt.title("Separabilite geometrique de la population en 3D via les features artisanales", fontsize=13, fontweight='bold', pad=20)
ax.view_init(elev=25, azim=45)
ax.legend(loc='upper left', fontsize=11)
plt.show()

# ==========================================
# 4. MODELISATION LINEAIRE ET EVALUATION
# ==========================================
print("\n" + "="*60)
print("ENTRAINEMENT DU MODELE DE REGRESSION LOGISTIQUE")
print("="*60)

features_selectionnees = ['ratio_ecran_sommeil', 'charge_mentale_index', 'seuil_synergie_critique']
X = df_ajuste[features_selectionnees]
y = df_ajuste['depression_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred_test = model.predict(X_test_scaled)

print("--- RAPPORT DE CLASSIFICATION (JEU DE TEST) ---")
print(classification_report(y_test, y_pred_test))

# Matrice de confusion
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sain (0)', 'Depressif (1)'])
disp.plot(cmap='Reds', ax=ax)
ax.set_title("Matrice de Confusion sur l'echantillon de Test", fontweight='bold')
plt.tight_layout()
plt.show()

feature_importance = pd.DataFrame({'Variable': features_selectionnees, 'Coefficient': model.coef_[0]})
print("\n--- IMPORTANCE DES VARIABLES ---")
print(feature_importance.sort_values(by='Coefficient', ascending=False))
