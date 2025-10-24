# 📚 AI Book Discovery Platform

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*Discover your next favorite book with AI-powered semantic search and intelligent analysis*

[✨ Features](#-features) • [🚀 Quick Start](#-quick-start) • [💡 How It Works](#-how-it-works) • [📖 Usage](#-usage) • [🛠️ Tech Stack](#️-tech-stack)

<img src="https://raw.githubusercontent.com/yourusername/ai-book-discovery/main/assets/demo.gif" alt="Demo" width="600"/>

</div>

---

## 🌟 Overview

**AI Book Discovery Platform** is an intelligent book recommendation system that understands natural language queries and provides deep, contextual book recommendations. Unlike traditional search systems that rely on keyword matching, this platform uses cutting-edge AI to understand the **semantic meaning** of your search and provides expert-level analysis for each recommendation.

### 🎯 What Makes It Special?

```
Traditional Search: "science fiction AI"
   → Returns books with those exact keywords

AI Search: "I'm looking for a sci-fi novel that explores AI consciousness"
   → Understands themes, context, and intent
   → Finds books about AI ethics, sentience, and philosophical questions
   → Provides expert analysis of why each book matches your needs
```

---

## ✨ Features

### 🔍 **Semantic Search Engine**
- Natural language understanding powered by `nomic-embed-text`
- No need for exact keyword matching
- Searches by meaning, theme, and context

### 🧠 **AI-Powered Analysis**
- **Query Analysis**: Understand your reading preferences and search intent
- **Expert Book Reviews**: Get AI-generated summaries, themes, and audience insights
- **Reader Profiling**: Discover what type of reader you are based on your searches

### 📊 **Flexible Data Management**
- Default curated book dataset included
- Upload your own custom CSV datasets
- Support for any book collection with titles and descriptions

### 🏠 **100% Local & Private**
- Runs entirely on your machine using Ollama
- No data sent to external APIs
- Complete privacy and control

### ⚡ **Real-time Processing**
- Instant semantic search results
- Dynamic AI analysis generation
- Responsive Streamlit interface

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Ollama** running locally ([Install Ollama](https://ollama.ai))
- Required AI models pulled

### Step 1: Install Ollama & Pull Models

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

### Step 2: Clone & Install

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-book-discovery.git
cd ai-book-discovery

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Launch the App

```bash
streamlit run OllamaLLM.py
```

The app will open in your browser at `http://localhost:8501` 🎉

---

## 💡 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   User Query Input                      │
│         "I want a fantasy book about dragons"           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Embedding Generation                       │
│           (nomic-embed-text via Ollama)                 │
│  Query → [0.234, -0.891, 0.456, ...] (vector)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Semantic Similarity Search                   │
│         Compare query vector with all books             │
│         Using Cosine Similarity Algorithm               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Top Matches Retrieved                      │
│    Books ranked by semantic relevance score             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           AI Analysis Generation                        │
│          (llama3.2:1b via Ollama)                       │
│  • Reader Profile    • Search Intent                    │
│  • Collection Themes • Individual Book Analysis         │
└─────────────────────────────────────────────────────────┘
```

### The Magic Behind Semantic Search

**Traditional Keyword Search:**
```python
if "dragon" in book_description:
    return book  # Simple string matching
```

**AI Semantic Search:**
```python
# Understands meaning and context
query_embedding = embed("I want a fantasy book about dragons")
book_embedding = embed("Epic tale featuring mythical creatures and fire-breathing beasts")

similarity = cosine_similarity(query_embedding, book_embedding)
# High similarity even without exact word "dragon"!
```

---

## 📖 Usage

### Basic Search

1. **Enter Your Query**: Type a natural language description of what you're looking for

```
Examples:
- "A thriller that keeps me on the edge of my seat"
- "Books about personal growth and mindfulness"
- "Historical fiction set during World War II"
- "Funny books that will make me laugh out loud"
```

2. **Get Recommendations**: View semantically matched books with relevance scores

3. **Explore AI Analysis**: 
   - Read the comprehensive query analysis
   - Expand individual books for expert insights

### Using Custom Datasets

1. **Prepare Your CSV**: Ensure it has at least these columns:
   - `Title` (required)
   - `Description` (required)
   - Optional: `Authors`, `Category`, `Price`, etc.

2. **Upload**: Use the sidebar file uploader

3. **Search**: Your custom books are now searchable!

### Example Workflow

```
🔍 User Query:
"I need a book that helps me understand human psychology"

🤖 AI Understanding:
- Intent: Educational self-improvement
- Themes: Psychology, human behavior, mental processes
- Reader Profile: Curious learner seeking knowledge

📚 Top Recommendations:
1. "Thinking, Fast and Slow" (95% match)
   Expert Analysis:
   • Summary: Explores dual-process theory of mind
   • Themes: Cognitive biases, decision-making, rationality
   • Audience: Psychology enthusiasts, professionals

2. "The Power of Habit" (92% match)
   Expert Analysis:
   • Summary: Examines habit formation and behavioral change
   • Themes: Neuroscience, productivity, self-improvement
   • Audience: Self-help readers, behavioral scientists
```

---

## 🛠️ Tech Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **LLM Runtime** | Ollama | Local AI model execution |
| **Embedding Model** | nomic-embed-text | Semantic understanding |
| **Generative Model** | llama3.2:1b | Text analysis & insights |
| **Data Processing** | Pandas | Dataset manipulation |
| **ML Algorithms** | Scikit-learn | Similarity calculations |

### AI Models Explained

#### 🔷 nomic-embed-text
- **Purpose**: Convert text into numerical vectors (embeddings)
- **Size**: ~274MB
- **Use Case**: Semantic search and similarity matching
- **Speed**: Fast inference (~50ms per query)

#### 🔶 llama3.2:1b
- **Purpose**: Generate human-like analytical text
- **Size**: ~1.3GB
- **Use Case**: Query analysis and book insights
- **Speed**: Moderate (~2-3s per analysis)

---

## 📂 Project Structure

```
ai-book-discovery/
├── OllamaLLM.py              # Main application file
├── BooksDataset.csv          # Default book dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── assets/                   # Images and demos
│   └── demo.gif
└── .gitignore
```

---

## 🎨 Features in Detail

### 1. Semantic Search Engine

**How traditional search fails:**
```
Query: "Books about overcoming adversity"
Traditional: Searches for exact words "overcoming" AND "adversity"
Result: Misses books about "triumph", "resilience", "perseverance"
```

**How AI search succeeds:**
```
Query: "Books about overcoming adversity"
AI Understanding: Concepts of struggle, growth, resilience, triumph
Result: Finds all thematically related books, regardless of exact wording
```

### 2. Comprehensive Query Analysis

After each search, get insights into:

- **📊 Reader Profile**: What your search says about you
  ```
  "You appear to be seeking intellectually stimulating content 
   with practical applications in daily life."
  ```

- **🎯 Search Intent**: What you're really looking for
  ```
  "Intent: Self-improvement through understanding psychological 
   principles and human behavior patterns."
  ```

- **📚 Collection Themes**: Common threads in recommendations
  ```
  "This collection emphasizes cognitive science, behavioral 
   psychology, and evidence-based self-development."
  ```

### 3. Expert Book Analysis

Click on any recommended book to reveal:

- **✍️ Executive Summary**: Concise overview of the book's content
- **🎭 Key Themes**: Main topics and concepts explored
- **👥 Target Audience**: Who will benefit most from this book

---

## 🔧 Configuration

### Customizing AI Models

Edit the model names in `OllamaLLM.py`:

```python
# For embedding (search)
EMBEDDING_MODEL = "nomic-embed-text"  # Change to your preferred embedding model

# For text generation (analysis)
LLM_MODEL = "llama3.2:1b"  # Options: llama3.2:3b, mistral, etc.
```

### Adjusting Search Results

```python
# Number of recommendations to show
TOP_K = 5  # Change to show more or fewer results

# Minimum similarity threshold
MIN_SIMILARITY = 0.3  # Range: 0.0 to 1.0
```

---


## 🐛 Troubleshooting

### Common Issues

**Problem**: `Connection refused` error
```bash
# Solution: Ensure Ollama is running
ollama serve
```

**Problem**: Models not found
```bash
# Solution: Pull the required models
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

**Problem**: Slow performance
```bash
# Solution: Use a smaller LLM model
# In OllamaLLM.py, change to:
LLM_MODEL = "llama3.2:1b"  # Instead of larger models
```

**Problem**: Out of memory
```bash
# Solution: Close other applications or use quantized models
ollama pull llama3.2:1b-q4_0  # Quantized version uses less RAM
```

---

## 📊 Performance Metrics

| Operation | Time (avg) | Notes |
|-----------|------------|-------|
| Initial embedding generation | 2-5 min | One-time per dataset |
| Query embedding | ~50ms | Per search |
| Similarity search | ~100ms | For 10K books |
| AI analysis generation | 2-3s | Per query |
| Individual book analysis | 1-2s | On-demand |

**System Requirements:**
- RAM: 8GB minimum (16GB recommended)
- CPU: Modern multi-core processor
- Storage: ~2GB for models

---

## 💡 Use Cases

### For Readers
- Discover books based on mood or current interests
- Find books similar to ones you've enjoyed
- Explore new genres with AI guidance

### For Librarians
- Help patrons find books more effectively
- Manage and search large catalogs
- Provide personalized recommendations

### For Book Clubs
- Find books matching group preferences
- Generate discussion themes automatically
- Analyze book connections and themes

### For Researchers
- Search academic books by concept
- Find related literature quickly
- Analyze book themes and trends

---

## 🙏 Acknowledgments

- **Ollama Team** for making local LLMs accessible
- **Streamlit** for the amazing web framework
- **Nomic AI** for the excellent embedding model
- **Meta AI** for the Llama 3.2 model
- The open-source community for inspiration and support

---

## 🎓 Learn More

### Related Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Understanding Embeddings](https://www.pinecone.io/learn/vector-embeddings/)
- [Semantic Search Explained](https://www.elastic.co/what-is/semantic-search)

---
