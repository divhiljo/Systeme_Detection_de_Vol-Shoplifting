# 🔍 Système de Détection de Vol par Intelligence Artificielle

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Technologies utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Contributeurs](#contributeurs)
- [Licence](#licence)

---

## 📌 Aperçu du projet

Ce projet vise à développer un système intelligent et automatique capable des **comportement suspect et des actes de vol** en temps réel. Il analyse des flux vidéo pour identifier des activités suspectes et alerter les responsables de sécurité.

Il repose sur deux modèles d’intelligence artificielle :

Détection de personnes
Détection de comportements de vol

> **Cas d'usage :** Surveillance de magasins, entrepôts.

---

## ✨ Fonctionnalités

- 🎥 Détection en temps réel des personnes dans une scène
- 🧠 Détection de comportements suspects via des modèles IA
- 🚨 Système d'alerte automatique envoie d'email(notification)
- 📊 Mini tableau de bord de surveillance
- 🗂️ Enregistrement et archivage des incidents détectés
- 📈 Statistiques et rapports de sécurité

---

## 🛠️ Technologies utilisées  

| Technologie | Rôle |
|---|---|
| Python 3.13.7 | Langage principal |
| OpenCV | Traitement d'images et vidéos |
| Numpy | Modèles de deep learning |
| YOLO | Détection d'objets et de personnes |
| tkinter | Interface graphique |


---

## ⚙️ Installation

### Prérequis

- Python 3.13.7
- pip
- Git
- (Optionnel) Avoir un bon GPU possedant CUDA pour de meilleures performances

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/divhiljo/Systeme_Detection_de_Vol-Shoplifting.git
cd detection-de-vol

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows


## 🚀 Utilisation

# Lancer le système de détection
python main.py

---

## 🤝 Contributeurs

| Nom | Rôle |
|---|---|
| BEBENE MBABE Guy-durent | Développeur principal |
| MAFFOCK Junie-Varette | Chef d'equipe |
| NANKAP-NDIZEU Loic-Aurel | Mentor/Conseiller Technique |

---

## 📄 Licence

Ce projet est sous licence **Institut Ucac-Icam**.

---

## 📞 Contact

Pour toute question ou suggestion :
- 📧 Email : guydurentbebenembabe@gmail.com
- 💼 LinkedIn : www.linkedin.com/in/guy-durent-bebene-mbabe-576152341
- 🐙 GitHub : https://github.com/divhiljo

---

> ⚠️ **Avertissement :** Ce système est destiné à des usages légaux et éthiques uniquement. Assurez-vous de respecter les lois locales sur la surveillance et la protection des données personnelles.
