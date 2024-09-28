# Domain-Specific Learning Platform

## Problem Statement

Students and professionals face challenges in navigating vast amounts of unstructured information. There's a lack of personalized, domain-specific learning resources for effective preparation. This platform addresses this issue by offering an integrated solution that provides curated domain-specific articles, Retrieval-Augmented Generation (RAG) chatbots, quiz generators, and the ability to study interactively from personal documents. Additionally, it recommends relevant groups and communities based on user activity to foster networking.

## Solution Overview

This platform offers a suite of tools and features to enhance personalized learning and collaboration:

### 1. Domain-Specific Resources
- **Curated Articles**: Tailored learning materials focused on specific domains, such as:
  - Artificial Intelligence (AI) & Machine Learning (ML)
  - Web Development
  - App Development
  - Blockchain

### 2. RAG Chatbot
- **Retrieval-Augmented Generation (RAG)**: Provides real-time, domain-specific assistance. 
- **Answering User Queries**: Uses Qdrant and FAISS for efficient vector search and Langchain for generation-based responses.

### 3. Interactive Quiz Generator
- **Educhain Integration**: Automatically generates quizzes using Educhain’s QnA engine.
- **Personalized Learning**: Users can test and reinforce their knowledge in specific domains.

### 4. Play Quiz and Chat with Document
- **Document Upload**: Allows users to upload documents in PDF format and interact with them via the chatbot.
- **Personalized Quiz Generation**: Combines RAG and Educhain to create quizzes based on uploaded documents for better study support.

### 5. Community Recommendations
- **Collaborative Filtering**: Recommends groups and communities based on user activity (e.g., domain visits, engagement scores).
- **Networking Opportunities**: Enhances collaboration by suggesting like-minded communities.

---

## Technical Approach

### Frontend
- **Framework**: Built using React.js for a responsive and interactive user interface.

### Backend
- **Framework**: Python with Django or Flask to create core APIs and handle user requests.
- **RAG Chatbot**: Integrated using Langchain, Qdrant, and FAISS for vector storage and retrieval.

### Quiz Generator
- **Educhain**: Integrated Educhain’s QnA engine with RAG for dynamic quiz generation.

### Chat with Document
- **Document Parsing**: Uses libraries like PyPDF2 or Apache Tika for document parsing, and Tesseract OCR for non-text PDFs.
- **Integration**: Connects document parsing with the RAG chatbot for interactive study.

---

## Database

- **User Data and Content**: 
  - PostgreSQL for structured data such as user profiles, quizzes, and group/community information.
- **Vector Search**:
  - Qdrant for storing vectorized representations of documents and user queries.

---

## Recommendation Engine

- **Collaborative Filtering**: Based on user activity (stored in the `UserProfile` table in PostgreSQL), recommendations are generated for groups and communities.
  
---

## Deployment

- **Cloud Hosting**: AWS or Google Cloud Platform (GCP) for scalable cloud infrastructure.
- **Containerization**: Uses Docker for containerizing the application to ensure easy deployment and scalability.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Nidhish-714/StudySphere.git
   cd domain-learning-platform
   pythom -m venv myenv
   myenv\Scripts\activate
   pip install -r requirements.txt
