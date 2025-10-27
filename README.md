REPO IN DEVELOPMENT STAGE
# 🧠 REG2025 Challenge – Multimodal WSI-Report Matching using TITAN

## 📘 Project Overview

This repository contains our solution for the **REG2025 Challenge**, which focuses on **retrieving diagnostic reports that best match pathology Whole Slide Images (WSIs)**.  
The project leverages **multimodal learning** — combining image and text understanding — to align visual pathology data with clinical textual descriptions.

Our approach builds upon **TITAN**, a transformer-based model for joint vision–language representation, integrated through the **Trident framework**.

---

## 🏆 About the REG2025 Challenge

The **REG (Report–Image Generation) Challenge 2025** is an international research competition that targets **cross-modal retrieval** and **report generation** in the pathology domain.  
Participants are provided with a dataset of **Whole Slide Images (WSIs)** and corresponding **diagnostic reports**, with the goal of:

- Learning a joint embedding space for images and text.  
- Retrieving the most relevant diagnostic report for each unseen WSI.  
- Advancing multimodal medical understanding through AI.

---

## ❓ Problem Statement

Given a pathology WSI, the task is to **find the most relevant diagnostic report** among a set of existing reports.

This problem is challenging because:
- WSIs are extremely large and complex.
- Clinical reports contain rich domain-specific language.
- The relationship between image and text is **semantic**, not pixel-level.

To solve this, we apply **multimodal embedding learning**:
- Encode both modalities (image and text) into a **shared latent space**.
- Compute **similarity scores** (e.g., cosine similarity) between WSIs and reports.
- Retrieve the report with the highest similarity score for each WSI.

---

## 🧩 Our Approach

### 🔹 Overview

We designed a **retrieval-based system** that operates in three main stages:

1. **Extract WSI Patch Features:**  
   Using pre-computed patch embeddings from TITAN’s Vision Encoder.

2. **Aggregate Patch Features into Slide Embeddings:**  
   TITAN’s `encode_slide_from_patch_features()` method combines patch-level features into a single slide-level embedding.

3. **Extract Report Embeddings:**  
   Each diagnostic report is encoded using TITAN’s Text Encoder to obtain textual feature vectors.

4. **Similarity Search:**  
   - Normalize both embeddings.  
   - Compute cosine similarity between each test image embedding and all training report embeddings.  
   - Assign the report with the highest similarity as the prediction.

---

### 🧮 Pseudocode Summary

```python
# Step 1: Load embeddings
train_ids, train_embs, train_reports = load_train_data()
test_ids, test_embs = load_test_embeddings()

# Step 2: Project to shared embedding space
train_embs = F.normalize(train_embs @ model.text_encoder.proj, dim=-1)
test_embs = F.normalize(test_embs @ model.vision_encoder.proj, dim=-1)

# Step 3: Similarity search
for test_emb in test_embs:
    similarities = torch.mm(test_emb.unsqueeze(0), train_embs.T)
    best_match = torch.argmax(similarities)
    matched_report = train_reports[train_ids[best_match]]
