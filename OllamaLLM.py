import pandas as pd
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
from datetime import datetime
import json
import io
import re

st.set_page_config(
    page_title="Enterprise AI Book Discovery",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: #000000;
            padding: 0;
        }
        
        [data-testid="stSidebar"] {
            background: #0a0a0a;
            border-right: 1px solid #1a1a1a;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
            padding: 3rem 2rem;
            text-align: center;
            border-bottom: 1px solid #2a2a2a;
        }
        
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            color: #888888;
            margin-bottom: 1.5rem;
        }
        
        .ai-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            margin: 0.3rem;
        }
        
        .content-section {
            background: #000000;
            padding: 2rem;
        }
        
        .input-card {
            background: #0f0f0f;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
            margin: 1.5rem 0;
        }
        
        .book-card {
            background: #0f0f0f;
            padding: 1.8rem;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .book-card:hover {
            background: #1a1a1a;
            border-color: #4F46E5;
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.2);
        }
        
        .book-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #4F46E5;
            margin-bottom: 0.8rem;
        }
        
        .book-meta {
            color: #cccccc;
            font-size: 0.95rem;
            margin: 0.4rem 0;
        }
        
        .similarity-badge {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0.5rem 0.5rem 0;
        }
        
        .price-badge {
            background: #10B981;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            display: inline-block;
        }
        
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #ffffff;
            margin: 2.5rem 0 1.5rem 0;
            padding-bottom: 0.8rem;
            border-bottom: 2px solid #4F46E5;
        }
        
        .stat-card {
            background: #0f0f0f;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #2a2a2a;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stat-label {
            color: #888888;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4);
        }
        
        .stTextArea textarea, .stTextInput input {
            background: #0a0a0a !important;
            border: 1px solid #2a2a2a !important;
            color: white !important;
            border-radius: 8px !important;
        }
        
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #4F46E5 !important;
        }
        
        .info-box {
            background: #0f0f0f;
            padding: 1.2rem;
            border-radius: 8px;
            border-left: 3px solid #4F46E5;
            color: #cccccc;
            margin: 1rem 0;
        }
        
        .success-box {
            background: #0f0f0f;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #10B981;
            color: #10B981;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: #0f0f0f;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #F59E0B;
            color: #F59E0B;
            margin: 1rem 0;
        }
        
        label, .stMarkdown p, .stMarkdown li {
            color: #cccccc !important;
        }
        
        h1, h2, h3 {
            color: #ffffff !important;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: #555555;
            border-top: 1px solid #1a1a1a;
            margin-top: 3rem;
        }
        
        [data-testid="stExpander"] {
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
        }
        
        .data-source-badge {
            background: #7C3AED;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
            margin-left: 0.5rem;
        }
        
        .analysis-container {
            background: #0a0a0a;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #4F46E5;
            margin: 1.5rem 0;
        }
        
        .analysis-section {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #4F46E5;
            margin: 1rem 0;
        }
        
        .section-title {
            color: #7C3AED;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        div[data-testid="stExpander"] div[role="button"] p {
            color: #4F46E5 !important;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv('BooksDataset.csv')
        df = df.dropna(subset=['Description'])
        df['Category'] = df['Category'].fillna('Uncategorized')
        df['Source'] = 'Default Library'
        if len(df) > 500:
            df = df.head(500)
        return df, None
    except Exception as e:
        return None, str(e)

def load_custom_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        required_columns = ['Title', 'Description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        df = df.dropna(subset=['Description'])
        
        if 'Authors' not in df.columns:
            df['Authors'] = 'Unknown'
        if 'Category' not in df.columns:
            df['Category'] = 'Uncategorized'
        if 'Price Starting With ($)' not in df.columns:
            df['Price Starting With ($)'] = 0.0
            
        df['Source'] = 'Custom Upload'
        
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def get_embeddings(text: str):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response['embedding']
    except Exception as e:
        st.error(f"Embedding generation error: {e}")
        return []

def generate_embeddings_batch(df):
    embeddings = []
    progress = st.progress(0)
    status = st.empty()
    
    total = len(df)
    for idx, desc in enumerate(df['Description']):
        embeddings.append(get_embeddings(desc))
        progress.progress((idx + 1) / total)
        status.markdown(f'<p style="color: #4F46E5;">AI Processing: {idx + 1}/{total} documents analyzed</p>', unsafe_allow_html=True)
    
    progress.empty()
    status.empty()
    return embeddings

def get_comprehensive_analysis(query: str, books: list) -> dict:
    books_summary = "\n".join([f"- {b['Title']} by {b['Authors']} ({b['Category']})" for b in books[:5]])
    
    prompt = f"""As a professional librarian AI assistant, analyze this search query and the recommended books.

Search Query: "{query}"

Top Recommended Books:
{books_summary}

Provide a comprehensive analysis. Answer each question directly in plain text without any formatting:

1. Reader Profile: Write 2-3 sentences describing this reader based on their search
2. Search Intent: Explain what the reader is looking for
3. Collection Theme: Describe the unifying theme across these recommendations
4. Why These Books: Explain specifically why these books match perfectly
5. Next Steps: Suggest reading directions or related topics to explore

Use plain text only. No HTML tags, no markdown, no special formatting."""
    
    try:
        response = ollama.generate(model="llama3.2:1b", prompt=prompt)
        text = response['response']
        
        text = re.sub(r'<[^>]+>', '', text)
        
        lines = text.strip().split('\n')
        result = {
            "reader_profile": "",
            "search_intent": "",
            "collection_theme": "",
            "why_these_books": "",
            "next_steps": ""
        }
        
        current_key = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'reader profile' in line.lower() or line.startswith('1'):
                current_key = "reader_profile"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'reader profile:?', '', line, flags=re.IGNORECASE).strip()
            elif 'search intent' in line.lower() or line.startswith('2'):
                current_key = "search_intent"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'search intent:?', '', line, flags=re.IGNORECASE).strip()
            elif 'collection theme' in line.lower() or line.startswith('3'):
                current_key = "collection_theme"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'collection theme:?', '', line, flags=re.IGNORECASE).strip()
            elif 'why these books' in line.lower() or line.startswith('4'):
                current_key = "why_these_books"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'why these books:?', '', line, flags=re.IGNORECASE).strip()
            elif 'next steps' in line.lower() or line.startswith('5'):
                current_key = "next_steps"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'next steps:?', '', line, flags=re.IGNORECASE).strip()
            
            if current_key and line:
                if result[current_key]:
                    result[current_key] += " " + line
                else:
                    result[current_key] = line
        
        if not any(result.values()):
            return {
                "reader_profile": "A reader seeking quality literature in this domain.",
                "search_intent": "To discover well-crafted books matching specific interests.",
                "collection_theme": "Curated selection based on semantic similarity.",
                "why_these_books": "These books share thematic and stylistic elements aligned with your query.",
                "next_steps": "Explore related categories and authors in this genre."
            }
        
        return result
        
    except Exception as e:
        return {
            "reader_profile": "A reader seeking quality literature in this domain.",
            "search_intent": "To discover well-crafted books matching specific interests.",
            "collection_theme": "Curated selection based on semantic similarity.",
            "why_these_books": "These books share thematic and stylistic elements aligned with your query.",
            "next_steps": "Explore related categories and authors in this genre."
        }

def get_detailed_book_analysis(title: str, description: str, category: str, authors: str) -> dict:
    prompt = f"""As a professional book reviewer AI, provide a detailed analysis of this book.

Title: {title}
Authors: {authors}
Category: {category}
Description: {description}

Answer each question directly in plain text without any formatting:

1. Executive Summary: Write 2-3 sentences summarizing the book
2. Key Themes: List 5-7 main themes, comma-separated
3. Target Audience: Describe who would benefit from this book
4. Unique Value: Explain what makes this book stand out
5. Reading Context: Describe best scenarios for reading this book

Use plain text only. No HTML tags, no markdown, no special formatting."""
    
    try:
        response = ollama.generate(model="llama3.2:1b", prompt=prompt)
        text = response['response']
        
        text = re.sub(r'<[^>]+>', '', text)
        
        lines = text.strip().split('\n')
        result = {
            "executive_summary": "",
            "key_themes": "",
            "target_audience": "",
            "unique_value": "",
            "reading_context": ""
        }
        
        current_key = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'executive summary' in line.lower() or line.startswith('1'):
                current_key = "executive_summary"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'executive summary:?', '', line, flags=re.IGNORECASE).strip()
            elif 'key themes' in line.lower() or line.startswith('2'):
                current_key = "key_themes"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'key themes:?', '', line, flags=re.IGNORECASE).strip()
            elif 'target audience' in line.lower() or line.startswith('3'):
                current_key = "target_audience"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'target audience:?', '', line, flags=re.IGNORECASE).strip()
            elif 'unique value' in line.lower() or line.startswith('4'):
                current_key = "unique_value"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'unique value:?', '', line, flags=re.IGNORECASE).strip()
            elif 'reading context' in line.lower() or line.startswith('5'):
                current_key = "reading_context"
                line = re.sub(r'^[\d\.]+\s*', '', line)
                line = re.sub(r'reading context:?', '', line, flags=re.IGNORECASE).strip()
            
            if current_key and line:
                if result[current_key]:
                    result[current_key] += " " + line
                else:
                    result[current_key] = line
        
        if not any(result.values()):
            return {
                "executive_summary": f"A comprehensive work in {category} offering valuable insights.",
                "key_themes": category,
                "target_audience": "Readers interested in this subject matter",
                "unique_value": "Well-researched content with clear presentation",
                "reading_context": "Academic study, professional development, or personal interest"
            }
        
        return result
        
    except Exception as e:
        return {
            "executive_summary": f"A comprehensive work in {category} offering valuable insights.",
            "key_themes": category,
            "target_audience": "Readers interested in this subject matter",
            "unique_value": "Well-researched content with clear presentation",
            "reading_context": "Academic study, professional development, or personal interest"
        }

def recommend_books(query: str, df: pd.DataFrame, top_n: int = 10):
    user_embedding = get_embeddings(query)
    
    if len(user_embedding) == 0:
        return pd.DataFrame()
    
    similarities = [
        cosine_similarity([user_embedding], [emb])[0][0]
        for emb in df['Description_embeddings']
    ]
    
    df['Similarity'] = similarities
    return df.nlargest(top_n, 'Similarity')

def display_book_card(book, idx):
    similarity_pct = book['Similarity'] * 100
    
    st.markdown(f"""
        <div class="book-card">
            <div class="book-title">
                {idx}. {book['Title']}
                <span class="data-source-badge">{book['Source']}</span>
            </div>
            <div class="book-meta"><strong>Author:</strong> {book['Authors']}</div>
            <div class="book-meta"><strong>Category:</strong> {book['Category']}</div>
            <div style="margin-top: 1rem;">
                <span class="similarity-badge">Relevance Score: {similarity_pct:.1f}%</span>
                <span class="price-badge">${book['Price Starting With ($)']:.2f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("View Full Description"):
            st.markdown(f'<div style="color: #cccccc;">{book["Description"]}</div>', unsafe_allow_html=True)
    
    with col2:
        with st.expander("AI Expert Analysis"):
            with st.spinner("AI analyzing book content..."):
                analysis = get_detailed_book_analysis(
                    book['Title'],
                    book['Description'],
                    book['Category'],
                    book['Authors']
                )
                
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                
                st.markdown('<p class="section-title">Executive Summary</p>', unsafe_allow_html=True)
                st.write(analysis['executive_summary'])
                
                st.markdown('<p class="section-title">Key Themes</p>', unsafe_allow_html=True)
                st.write(analysis['key_themes'])
                
                st.markdown('<p class="section-title">Target Audience</p>', unsafe_allow_html=True)
                st.write(analysis['target_audience'])
                
                st.markdown('<p class="section-title">Unique Value</p>', unsafe_allow_html=True)
                st.write(analysis['unique_value'])
                
                st.markdown('<p class="section-title">Reading Context</p>', unsafe_allow_html=True)
                st.write(analysis['reading_context'])
                
                st.markdown('</div>', unsafe_allow_html=True)

def main():
    apply_styles()
    
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">AI Book Discovery Platform</div>
            <div class="hero-subtitle">Intelligent Recommendation System Powered by Large Language Models (LLM)</div>
            <span class="ai-badge">Ollama LLM Engine</span>
            <span class="ai-badge">Neural Embeddings</span>
            <span class="ai-badge">Semantic Search</span>
            <span class="ai-badge">Custom Data Support</span>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Data Source Management")
        
        data_source = st.radio(
            "Select Data Source",
            ["Default Library Dataset", "Upload Custom Dataset"],
            help="Choose between default library or upload your own book collection"
        )
        
        st.markdown("---")
        
        if data_source == "Upload Custom Dataset":
            st.markdown("### Upload Your Dataset")
            st.markdown("""
            <div class="info-box">
            <strong>Required Format:</strong><br>
            CSV file with columns:<br>
            • Title (required)<br>
            • Description (required)<br>
            • Authors (optional)<br>
            • Category (optional)<br>
            • Price Starting With ($) (optional)
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                df, error = load_custom_data(uploaded_file)
                if error:
                    st.markdown(f'<div class="warning-box">{error}</div>', unsafe_allow_html=True)
                    df = None
                else:
                    st.markdown(f'<div class="success-box">Successfully loaded {len(df)} books from custom dataset</div>', unsafe_allow_html=True)
                    st.session_state.custom_df = df
                    st.session_state.use_custom = True
            else:
                if 'use_custom' in st.session_state:
                    del st.session_state.use_custom
        else:
            if 'use_custom' in st.session_state:
                del st.session_state.use_custom
        
        st.markdown("---")
        st.markdown("### Search Configuration")
        
        num_results = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            help="How many book recommendations to display"
        )
        
        st.markdown("---")
        st.markdown("### System Information")
        
        if 'embeddings_ready' in st.session_state:
            df_display = st.session_state.df
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df_display)}</div>
                <div class="stat-label">Books Indexed</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{df_display['Category'].nunique()}</div>
                <div class="stat-label">Categories</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">Active</div>
                <div class="stat-label">AI Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    if 'embeddings_ready' not in st.session_state or ('use_custom' in st.session_state and 'custom_df' in st.session_state and st.session_state.df.equals(st.session_state.custom_df) == False):
        with st.spinner("Initializing AI system and processing book database..."):
            if 'use_custom' in st.session_state and 'custom_df' in st.session_state:
                df = st.session_state.custom_df
            else:
                df, error = load_default_data()
                if error:
                    st.error(f"Error loading default data: {error}")
                    return
            
            df['Description_embeddings'] = generate_embeddings_batch(df)
            st.session_state.df = df
            st.session_state.embeddings_ready = True
            st.markdown('<div class="success-box">AI system initialized successfully</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(df)}</div><div class="stat-label">Total Books</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{df["Category"].nunique()}</div><div class="stat-label">Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(df["Authors"].unique())}</div><div class="stat-label">Authors</div></div>', unsafe_allow_html=True)
    with col4:
        source_type = "Custom" if 'use_custom' in st.session_state else "Default"
        st.markdown(f'<div class="stat-card"><div class="stat-number">{source_type}</div><div class="stat-label">Data Source</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">AI-Powered Search</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    st.markdown('<p style="color: #ffffff; font-size: 1.1rem; margin-bottom: 1rem;">Describe your reading preferences in natural language. Be as detailed as possible.</p>', unsafe_allow_html=True)
    
    query = st.text_area(
        "Search Query",
        placeholder="Example: I'm looking for a thought-provoking science fiction novel that explores artificial intelligence and human consciousness, with complex characters and philosophical depth...",
        height=120,
        label_visibility="collapsed"
    )
    
    search_btn = st.button("Search with AI", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if search_btn and query:
        with st.spinner("AI is analyzing your request against the entire library collection..."):
            recommendations = recommend_books(query, df, num_results)
            
            if not recommendations.empty:
                st.markdown('<div class="section-header">Comprehensive AI Analysis</div>', unsafe_allow_html=True)
                
                with st.spinner("Generating personalized insights using LLM..."):
                    books_data = recommendations[['Title', 'Authors', 'Category', 'Description']].head(5).to_dict('records')
                    analysis = get_comprehensive_analysis(query, books_data)
                    
                    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #4F46E5; margin-bottom: 1rem;">AI Librarian Analysis</h4>', unsafe_allow_html=True)
                    
                    st.markdown('<p class="section-title">Reader Profile</p>', unsafe_allow_html=True)
                    st.write(analysis['reader_profile'])
                    
                    st.markdown('<p class="section-title">Search Intent</p>', unsafe_allow_html=True)
                    st.write(analysis['search_intent'])
                    
                    st.markdown('<p class="section-title">Collection Theme</p>', unsafe_allow_html=True)
                    st.write(analysis['collection_theme'])
                    
                    st.markdown('<p class="section-title">Why These Books</p>', unsafe_allow_html=True)
                    st.write(analysis['why_these_books'])
                    
                    st.markdown('<p class="section-title">Recommended Next Steps</p>', unsafe_allow_html=True)
                    st.write(analysis['next_steps'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="section-header">Recommended Books ({len(recommendations)} Results)</div>', unsafe_allow_html=True)
                
                for idx, (_, book) in enumerate(recommendations.iterrows(), 1):
                    display_book_card(book, idx)
                
                st.markdown('<div class="section-header">Export & Share</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = recommendations[['Title', 'Authors', 'Category', 'Price Starting With ($)', 'Similarity', 'Source']].to_csv(index=False)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name=f"library_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_data = recommendations[['Title', 'Authors', 'Description', 'Category', 'Similarity', 'Source']].to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON Data",
                        data=json_data,
                        file_name=f"library_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    report = f"""ENTERPRISE LIBRARY RECOMMENDATION REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Query: {query}
Results: {len(recommendations)}
Data Source: {recommendations['Source'].iloc[0]}

AI ANALYSIS
{'='*50}
Reader Profile: {analysis['reader_profile']}

Search Intent: {analysis['search_intent']}

Collection Theme: {analysis['collection_theme']}

Why These Books: {analysis['why_these_books']}

Next Steps: {analysis['next_steps']}

RECOMMENDED BOOKS
{'='*50}
"""
                    for idx, (_, book) in enumerate(recommendations.iterrows(), 1):
                        report += f"\n{idx}. {book['Title']}\n"
                        report += f"   Author: {book['Authors']}\n"
                        report += f"   Category: {book['Category']}\n"
                        report += f"   Relevance: {book['Similarity']*100:.1f}%\n"
                        report += f"   Price: ${book['Price Starting With ($)']:.2f}\n"
                        report += f"   Source: {book['Source']}\n"
                    
                    st.download_button(
                        label="Download Full Report",
                        data=report,
                        file_name=f"library_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="warning-box">No matches found. Please refine your search query or check your data source.</div>', unsafe_allow_html=True)
    
    elif search_btn:
        st.markdown('<div class="warning-box">Please enter a search query to get AI-powered recommendations.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer">
            <p><strong>Enterprise AI Book Discovery Platform</strong></p>
            <p>Powered by Ollama LLM (nomic-embed-text + llama3.2:1b)</p>
            <p>Developed by Syahril Arfian Almazril </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()