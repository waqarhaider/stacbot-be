# 📘 FastAPI Backend – STAC RAG ChatBot

This repository contains the FastAPI backend for our Retrieval-Augmented Generation (RAG) project.
Follow all steps below before to successfully run the backend.

---

## ✅ **1. Prerequisites & API Keys**

### **1.1 Qdrant API Key**

1. Register on **Qdrant Cloud** using your **university / department email**:
   👉 [https://cloud.qdrant.io/settings](https://cloud.qdrant.io/settings)
2. Create a **Cloud Management Key**.
3. Using a department email ensures we **reuse the same cluster and collections** for all developers and no need to recreate for every new developer.

**Reference (Image 1):**
<img width="1084" height="596" alt="image" src="https://github.com/user-attachments/assets/11118bcc-3f5b-4964-b983-2d1a5bf7fd82" />

---

### **1.2 OpenAI API Key**

1. Create an API key via the OpenAI API portal:
   👉 [https://platform.openai.com/settings/organization/api-keys](https://platform.openai.com/settings/organization/api-keys)
2. Request access from professor, as it is managed using department email.

---

### **1.3 Set Environment Variables**

Set both environment variables in **all notebooks**:

```bash
export OPENAI_API_KEY="your_key_here"
export QDRANT_API_KEY="your_key_here"
```

---

## ✅ **2. Configure Your Qdrant Client**

Set the Qdrant client URL to the **Endpoint** shown in your Qdrant cluster dashboard.

**Reference (Image 2):**
<img width="1343" height="576" alt="image" src="https://github.com/user-attachments/assets/f6ea30c5-b693-425b-a6af-47b288371116" />


Example inside Python:

```python
qdrant = QdrantClient(
    url="https://<YOUR-ENDPOINT>.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)
```

Once this is set, **all setup steps are complete**.

---

# 📂 3. Preparing Documents & Building Qdrant Collections

### **3.1 Download PDFs**

1. Log in to STAC report portal:
   👉 [https://docs.stacresearch.com/user/login](https://docs.stacresearch.com/user/login)
2. Download all PDFs for your target report type (e.g., **A2**, **M3**, etc.).
3. Place them in the correct folder inside the repo:
4. For access, you can ask from STAC colleagues.

```
stacA2pdfs/
stacM3pdfs/
```

**Reference (Image 3):**
<img width="351" height="114" alt="image" src="https://github.com/user-attachments/assets/378169c0-6306-4cb5-883a-347d9cef613c" />


---

## ⚙️ **4. Generate Embeddings & Save to Qdrant**

Run the notebook:

**`start_2_until_vector_saving_openAI.ipynb`**

This script will:

* Read all PDFs
* Generate embeddings using OpenAI
* Upload vectors to the appropriate Qdrant collection

⏱️ **Expected time:** ~30 minutes (varies with number of documents)

Once finished, all vectors created based on documents will be available in Qdrant cluster.

---

## 🗂️ **5. Create an Empty Feedback Collection**

After your main collection is created:

* Manually create **one empty collection** in Qdrant.
  This collection stores **feedback entries** for that specific report type. You can copy the code from start_2_until_vector_saving_openAI notebook.

Example name:

```
stacA2_offline_feedback
stacM3_offline_feedback
```

**Reference (Image 4):**
<img width="1061" height="229" alt="image" src="https://github.com/user-attachments/assets/fd395ec0-afa0-40be-bb72-e174478ef11b" />

---

# ▶️ **6. Run the FastAPI Backend Locally**

To start the backend server on `localhost:8000`:

1. Navigate to the directory where **app.py** is located.
2. Run the following command:

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend will now be available at:

👉 **[http://localhost:8000](http://localhost:8000)**


# 🎉 Setup Complete!

You can now:

* Access the backend from frontend
* Start experimenting
* Develop new features

If you need any help setting it up, just let me know on waqarhaidercheema@gmail.com!
