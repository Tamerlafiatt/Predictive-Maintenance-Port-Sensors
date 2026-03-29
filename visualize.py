"""
visualize.py — EDA and model analysis plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("donnees_tiers.xlsx")
df["Date D.I"] = pd.to_datetime(df["Date D.I"], format="%d/%m/%Y %I:%M %p", dayfirst=True)
df["Date debut travail"] = pd.to_datetime(df["Date debut travail"], format="%d/%m/%Y %I:%M %p", dayfirst=True)
df["Date fin de travail"] = pd.to_datetime(df["Date fin de travail"], format="%d/%m/%Y %I:%M %p", dayfirst=True)
df["Durée calculée (h)"] = (df["Date fin de travail"] - df["Date debut travail"]).dt.total_seconds() / 3600
df["Mois"] = df["Date D.I"].dt.month

# 1. Distribution of repair hours
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["H. réalisées"].dropna(), bins=30, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title("Distribution des heures réalisées"); axes[0].set_xlabel("Heures")

# 2. Declared vs calculated duration
sns.scatterplot(x="H. réalisées", y="Durée calculée (h)", data=df, ax=axes[1], alpha=0.4)
axes[1].set_title("Durée déclarée vs calculée")
plt.tight_layout(); plt.savefig("results/duration_analysis.png", dpi=130); plt.close()

# 3. Interventions by equipment
plt.figure(figsize=(12, 5))
df["Libellé Equipement"].value_counts().plot(kind='bar', color='steelblue')
plt.title("Interventions par équipement"); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig("results/interventions_by_equipment.png", dpi=130); plt.close()

# 4. Avg repair hours by equipment
plt.figure(figsize=(12, 5))
sns.barplot(x="Libellé Equipement", y="H. réalisées", data=df, palette='Blues_d')
plt.title("Durée moyenne de réparation par équipement"); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig("results/avg_hours_by_equipment.png", dpi=130); plt.close()

# 5. Fault type distribution (pie)
labels = df["Titre D.I"].value_counts().index.tolist()
sizes  = df["Titre D.I"].value_counts().values
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Distribution des types de pannes"); plt.tight_layout()
plt.savefig("results/fault_distribution.png", dpi=130); plt.close()

print("Plots saved to results/")
