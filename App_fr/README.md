# Langchain RAGBot

**Langchain RAGBot** est une application interactive **Streamlit** qui permet aux utilisateurs de poser des questions relatives au contenu de leurs documents en utilisant un pipeline de **Retrieval-Augmented Generation (RAG)** basé sur des **LLM**.

## 📌 Aperçu

Il s’agit d’un **RAG conversationnel** construit entièrement à partir de **modèles LLM et de modèles d’embeddings open source**, pouvant être facilement étendu à des modèles plus puissants lorsque les ressources matérielles le permettent. L’objectif principal de cette application RAG est la **construction d’un système de question-réponse relativement au contenu de documents en langue française**. Ce modèle peut également être utilisé avec des documents en **langue anglaise**.

L’application propose :
- La possibilité de téléverser des documents  
- Une interface conversationnelle pour interroger les documents  
- La sélection du modèle et des embeddings (modèles open source provenant de Hugging Face et d’Ollama)  
- Un accès transparent au contexte récupéré utilisé pour générer les réponses  
---

## 🚀 Fonctionnalités

### **1. Plusieurs LLM**
- Llama3 (quantifié q4_K_M) 8b  
- Llama3 8b  
- Mistral 7b  

### **2. Plusieurs modèles d’embeddings**
- Paraphrase (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  
- Multilingual-e5 (intfloat/multilingual-e5-large)  
- Embedding français (dangvantuan/french-document-embedding)  

### **3. Téléversement de documents**
- Téléverser des documents et les interroger instantanément  

### **4. Interface de chat interactive**
- Historique de conversations sauvegardé dans l’état de session Streamlit  

### **5. Réponses en streaming**
- Les réponses de l’IA sont diffusées token par token pour une meilleure expérience par l'utilisateur  

### **6. Inspection du contexte**
- Visualisation des passages pertinents des documents récupérés pour chaque question  

---

## 📁 Structure du projet
```text
Langchain_RAGbot/App_fr
│
├── app.py                   # Application Streamlit principale
├── utils.py                 # Fonctions utilitaires (création de l’uploader de documents,
│                            # du récupérateur, de la chaîne RAG, etc.)
├── streamlit_functions.py   # Fonctions utilitaires pour l’application Streamlit
│                            # (affichage de l’uploader, messages du chat, etc.)
├── create_db.py             # Script pour la création de la base de données Chroma
├── requirement.txt          # Fichier contenant les dépendances requises
│
├── Chroma_store/            # Dossier contenant la base de données vectorielle
├── Documents/               # Dossier contenant les fichiers de documents
│
├── images/                  # Images enregistrées : figures, icônes, captures d’écran
│   └── example.png
│
└── README.md                # Documentation
```

## 🛠️ Installation

### ⚠️ Prérequis
Avant de lancer l’application, assurez-vous que les éléments suivants sont installés :
- Ollama est installé et fonctionne sur votre système  
- Les modèles LLM requis sont téléchargés dans Ollama  
- Les modèles d’embeddings requis sont téléchargés et disponibles  

L’application ne fonctionnera pas correctement si ces composants ne sont pas installés au préalable.

### 📥 Cloner le dépôt
```bash
git clone https://github.com/gitsujay25/Langchain_RAGbot.git
cd Langchain_RAGbot/App_fr
conda create -n langchain_rag python=3.10
conda activate langchain_rag
pip install -r requirements.txt
```
### ▶️ Lancer l’application

```bash
streamlit run app.py
```
Le tableau de bord s’ouvrira automatiquement dans votre navigateur à l’adresse : http://localhost:8501/

## 🏗️ Disposition de l’application

- **Barre latérale**
  - Sélection du modèle LLM  
  - Sélection du modèle d’embedding  
  - Téléversement de documents  
  - Option pour créer ou reconstruire le RAG  

- **Panneau central**
  - Interface de chat  
  - Saisie utilisateur et réponses de l’IA  

- **Panneau gauche**
  - Contexte récupéré lié à la requête de l’utilisateur

---

## ❓ Comment l’utiliser

1. Sélectionnez un **modèle LLM** dans la barre latérale  
2. Sélectionnez un **modèle d’embedding**  
3. Téléversez un ou plusieurs **documents**  
4. Appuyez sur le bouton **Build RAG** (ou **Rebuild RAG** – lors de la reconstruction du RAG, toutes les conversations précédentes seront supprimées et l’application recommencera à zéro)  
5. Posez une question via l’interface de chat  
6. Recevez une **réponse IA en streaming**  
7. Consultez le **contexte récupéré** dans le panneau de gauche

---

## 🧰 Conseils pour le développement

- Conservez les fonctions réutilisables dans `utils.py`  
- Utilisez `streamlit_functions.py` pour les fonctions de construction de l’interface Streamlit  
- Verrouillez les versions dans `requirements.txt` pour assurer la reproductibilité

---

## 🤝 Contribution

Les contributions sont les bienvenues !  
N’hésitez pas à forker le dépôt, ouvrir des issues ou soumettre des pull requests.

## 📬 Contact
Pour toute question ou suggestion :  
- Auteur : Sujay Ray  
- GitHub : https://github.com/gitsujay25  
- LinkedIn : https://www.linkedin.com/in/sujayray92/