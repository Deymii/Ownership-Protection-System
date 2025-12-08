import streamlit as st
import pandas as pd
import hashlib
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Initialize the semantic model (MiniLM)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# File paths
OWNERSHIP_FILE = 'ownership_records.csv'
CONTENT_FILE = 'registered_content.csv'

# Initialize CSV files if they don't exist
def initialize_files():
    if not os.path.exists(OWNERSHIP_FILE):
        df = pd.DataFrame(columns=['NFT_Hash', 'Owner', 'Title', 'Timestamp', 'Content_Hash'])
        df.to_csv(OWNERSHIP_FILE, index=False)
    
    if not os.path.exists(CONTENT_FILE):
        df = pd.DataFrame(columns=['Content_Hash', 'Content', 'Embedding', 'Timestamp'])
        df.to_csv(CONTENT_FILE, index=False)

initialize_files()

# Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation for semantic meaning
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

# Generate content hash
def generate_hash(text, owner):
    """Generate unique hash for content"""
    timestamp = datetime.now().isoformat()
    data = f"{text}{owner}{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()

# Compute semantic similarity
def compute_similarity(new_text, existing_embeddings):
    """Calculate cosine similarity between new text and existing content"""
    new_embedding = model.encode([new_text])
    
    if len(existing_embeddings) == 0:
        return 0.0, -1
    
    similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)
    
    return max_similarity, max_index

# Check originality
def check_originality(text, threshold=0.85):
    """Check if text is original by comparing with registered content"""
    processed_text = preprocess_text(text)
    
    # Load existing content
    if os.path.exists(CONTENT_FILE):
        content_df = pd.read_csv(CONTENT_FILE)
        
        if len(content_df) > 0:
            # Get embeddings from existing content
            existing_embeddings = []
            for idx, row in content_df.iterrows():
                embedding = model.encode([row['Content']])
                existing_embeddings.append(embedding[0])
            
            existing_embeddings = np.array(existing_embeddings)
            
            # Calculate similarity
            similarity, match_index = compute_similarity(processed_text, existing_embeddings)
            
            if similarity >= threshold:
                matched_content = content_df.iloc[match_index]
                return False, similarity, matched_content
            else:
                # Return the actual similarity even for original content
                return True, similarity, None
            
    return True, 0.0, None  # Only 0.0 if database is empty

# Mint NFT (generate hash and store)
def mint_nft(text, owner, title):
    """Create NFT record for original content"""
    content_hash = generate_hash(text, owner)
    nft_hash = hashlib.sha256(f"NFT_{content_hash}".encode()).hexdigest()
    timestamp = datetime.now().isoformat()
    
    # Store ownership record
    ownership_record = {
        'NFT_Hash': nft_hash,
        'Owner': owner,
        'Title': title,
        'Timestamp': timestamp,
        'Content_Hash': content_hash
    }
    
    ownership_df = pd.read_csv(OWNERSHIP_FILE)
    ownership_df = pd.concat([ownership_df, pd.DataFrame([ownership_record])], ignore_index=True)
    ownership_df.to_csv(OWNERSHIP_FILE, index=False)
    
    # Store content with embedding
    processed_text = preprocess_text(text)
    embedding = model.encode([processed_text])[0].tolist()
    
    content_record = {
        'Content_Hash': content_hash,
        'Content': processed_text,
        'Embedding': str(embedding),
        'Timestamp': timestamp
    }
    
    content_df = pd.read_csv(CONTENT_FILE)
    content_df = pd.concat([content_df, pd.DataFrame([content_record])], ignore_index=True)
    content_df.to_csv(CONTENT_FILE, index=False)
    
    return nft_hash, content_hash

# Streamlit UI
st.set_page_config(page_title="Digital Ownership Protection", page_icon="🔒", layout="wide")

st.title("🔒 Digital Ownership Protection System")
st.markdown("### Protecting Digital Ownership Using NFTs, Blockchain, and Semantic Similarity Analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Submit Content", "View Ownership Records", "System Statistics"])

# Similarity threshold setting
threshold = st.sidebar.slider("Similarity Threshold", 0.5, 1.0, 0.85, 0.05)
st.sidebar.info(f"Content with similarity ≥ {threshold:.0%} will be rejected")

if page == "Submit Content":
    st.header("📝 Submit Your Content")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        owner_name = st.text_input("Your Name", placeholder="Enter your name")
        content_title = st.text_input("Content Title", placeholder="Enter a title for your content")
        content_text = st.text_area("Content", placeholder="Paste or type your content here...", height=300)
        
        submit_button = st.button("🔍 Check Originality & Mint NFT", type="primary")
    
    with col2:
        st.info("**How it works:**\n\n"
                "1. Enter your details\n"
                "2. Submit your content\n"
                "3. System checks originality\n"
                "4. If original, NFT is minted\n"
                "5. Ownership is recorded")
    
    if submit_button:
        if not owner_name or not content_title or not content_text:
            st.error("⚠️ Please fill in all fields!")
        elif len(content_text.split()) < 10:
            st.warning("⚠️ Content is too short. Please provide at least 10 words.")
        else:
            with st.spinner("Analyzing content originality..."):
                is_original, similarity_score, matched_content = check_originality(content_text, threshold)
                
                if is_original:
                    st.success(f"✅ Content is original! (Similarity: {similarity_score:.2%})")
                    
                    with st.spinner("Minting NFT..."):
                        nft_hash, content_hash = mint_nft(content_text, owner_name, content_title)
                        
                    st.balloons()
                    st.success("🎉 NFT Successfully Minted!")
                    
                    st.markdown("### 📜 Your Ownership Certificate")
                    cert_col1, cert_col2 = st.columns(2)
                    
                    with cert_col1:
                        st.markdown(f"**NFT Hash:**")
                        st.code(nft_hash, language=None)
                        st.markdown(f"**Content Hash:**")
                        st.code(content_hash, language=None)
                    
                    with cert_col2:
                        st.markdown(f"**Owner:** {owner_name}")
                        st.markdown(f"**Title:** {content_title}")
                        st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                else:
                    st.error(f"❌ Content is NOT original! (Similarity: {similarity_score:.2%})")
                    st.warning("This content is too similar to existing registered content.")
                    
                    if matched_content is not None:
                        with st.expander("View Similar Content Details"):
                            st.markdown(f"**Matched Content Hash:** {matched_content['Content_Hash']}")
                            st.markdown(f"**Registered on:** {matched_content['Timestamp']}")

elif page == "View Ownership Records":
    st.header("📋 Ownership Records")
    
    if os.path.exists(OWNERSHIP_FILE):
        ownership_df = pd.read_csv(OWNERSHIP_FILE)
        
        if len(ownership_df) > 0:
            st.markdown(f"**Total Registered Content:** {len(ownership_df)}")
            
            # Search functionality
            search_term = st.text_input("🔍 Search by Owner or Title", placeholder="Enter search term...")
            
            if search_term:
                filtered_df = ownership_df[
                    ownership_df['Owner'].str.contains(search_term, case=False, na=False) |
                    ownership_df['Title'].str.contains(search_term, case=False, na=False)
                ]
            else:
                filtered_df = ownership_df
            
            # Display records
            for idx, row in filtered_df.iterrows():
                with st.expander(f"📄 {row['Title']} - by {row['Owner']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**NFT Hash:**")
                        st.code(row['NFT_Hash'], language=None)
                    with col2:
                        st.markdown(f"**Owner:** {row['Owner']}")
                        st.markdown(f"**Timestamp:** {row['Timestamp']}")
                        st.markdown(f"**Content Hash:** `{row['Content_Hash'][:16]}...`")
        else:
            st.info("No ownership records found. Submit content to get started!")
    else:
        st.info("No ownership records found. Submit content to get started!")

elif page == "System Statistics":
    st.header("📊 System Statistics")
    
    if os.path.exists(OWNERSHIP_FILE) and os.path.exists(CONTENT_FILE):
        ownership_df = pd.read_csv(OWNERSHIP_FILE)
        content_df = pd.read_csv(CONTENT_FILE)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Registered Content", len(ownership_df))
        
        with col2:
            unique_owners = ownership_df['Owner'].nunique() if len(ownership_df) > 0 else 0
            st.metric("Unique Owners", unique_owners)
        
        with col3:
            total_size = len(content_df)
            st.metric("Database Size", f"{total_size} records")
        
        if len(ownership_df) > 0:
            st.markdown("### Recent Registrations")
            recent_df = ownership_df.sort_values('Timestamp', ascending=False).head(5)
            st.dataframe(recent_df[['Title', 'Owner', 'Timestamp']], use_container_width=True)
        
        st.markdown("### System Information")
        st.info(f"**Model:** MiniLM (all-MiniLM-L6-v2)\n\n"
                f"**Similarity Algorithm:** Cosine Similarity\n\n"
                f"**Blockchain Simulation:** CSV-based storage\n\n"
                f"**Current Threshold:** {threshold:.0%}")
    else:
        st.info("No data available yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed by Zeyad Sabri Ali & Abdalrahman Muhammad Ali Alahmad</p>
    <p>Supervised by Dr. Abed Alanazi | Prince Sattam bin Abdulaziz University</p>
</div>
""", unsafe_allow_html=True)