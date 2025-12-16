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

# Custom CSS for dark theme and styled navigation
st.markdown("""
<style>
    /* Main background and sidebar */
    .stApp {
        background-color: #181A18;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1f2320;
    }
    
    /* Hide Streamlit's default header content but keep sidebar toggle */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
    }
    
    /* Keep sidebar collapse button visible */
    [data-testid="stSidebarCollapseButton"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Ensure collapsed sidebar button is visible */
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
    
    footer {
        display: none !important;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    /* Hide default radio button styling */
    [data-testid="stSidebar"] .stRadio > div {
        display: none;
    }
    
    /* Remove sidebar content padding for full-width buttons */
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] > div > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] .stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Sidebar button styling - full width, flat design */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 18px 25px !important;
        margin: 0 !important;
        text-align: left !important;
        font-size: 15px;
        transition: all 0.2s ease;
    }
    
    /* Secondary (unselected) buttons */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: #888888;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: #cccccc;
    }
    
    /* Primary (selected) buttons - subtle white */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border-left: 3px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Threshold slider styling */
    .threshold-container {
        background-color: #2a2d2a;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4444;
        margin: 10px 0;
    }
    
    .threshold-label {
        color: #ff4444;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with Navigation header
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("---")

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Navigation with styled buttons
pages = ["Dashboard", "Submit Content", "View Records"]

for page_name in pages:
    # Determine button style based on current page
    is_active = st.session_state.current_page == page_name
    
    if st.sidebar.button(
        f"{'📊' if page_name == 'Dashboard' else '📝' if page_name == 'Submit Content' else '📋'} {page_name}",
        key=f"nav_{page_name}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.current_page = page_name
        st.rerun()

page = st.session_state.current_page

# Default threshold value
threshold = 0.85

if page == "Submit Content":
    st.markdown("<h1 style='text-align: center;'>&#128274; Digital Ownership Protection System</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered form container
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        owner_name = st.text_input("Your Name", placeholder="Enter your name")
        content_title = st.text_input("Content Title", placeholder="Enter a title for your content")
        content_text = st.text_area("Content", placeholder="Paste or type your content here...", height=300)
        
        # Similarity threshold with red styling
        st.markdown("""
        <div class="threshold-container">
            <span class="threshold-label">⚠️ SIMILARITY THRESHOLD</span>
        </div>
        """, unsafe_allow_html=True)
        threshold = st.slider("Content with similarity above this threshold will be rejected", 0.5, 1.0, 0.85, 0.05)
        st.markdown(f"**Current threshold: {threshold:.0%}**")
        
        # Centered button
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        submit_button = st.button("🔍 Check Originality", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
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
                    
                    # Removed balloons effect
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

elif page == "View Records":
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

elif page == "Dashboard":
    st.header("📊 Dashboard")
    
    if os.path.exists(OWNERSHIP_FILE) and os.path.exists(CONTENT_FILE):
        ownership_df = pd.read_csv(OWNERSHIP_FILE)
        content_df = pd.read_csv(CONTENT_FILE)
        
        # Visual metrics with colored cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); padding: 20px; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">📄</h2>
                <h1 style="color: white; margin: 10px 0;">{}</h1>
                <p style="color: #E8F5E9; margin: 0;">Total Content</p>
            </div>
            """.format(len(ownership_df)), unsafe_allow_html=True)
        
        with col2:
            unique_owners = ownership_df['Owner'].nunique() if len(ownership_df) > 0 else 0
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2196F3, #1565C0); padding: 20px; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">👥</h2>
                <h1 style="color: white; margin: 10px 0;">{}</h1>
                <p style="color: #E3F2FD; margin: 0;">Unique Owners</p>
            </div>
            """.format(unique_owners), unsafe_allow_html=True)
        
        with col3:
            total_size = len(content_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF9800, #EF6C00); padding: 20px; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">💾</h2>
                <h1 style="color: white; margin: 10px 0;">{}</h1>
                <p style="color: #FFF3E0; margin: 0;">Database Records</p>
            </div>
            """.format(total_size), unsafe_allow_html=True)
        
        with col4:
            # Calculate today's registrations
            today = datetime.now().strftime('%Y-%m-%d')
            if len(ownership_df) > 0:
                today_count = len(ownership_df[ownership_df['Timestamp'].str.contains(today, na=False)])
            else:
                today_count = 0
            st.markdown("""
            <div style="background: linear-gradient(135deg, #9C27B0, #6A1B9A); padding: 20px; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">📅</h2>
                <h1 style="color: white; margin: 10px 0;">{}</h1>
                <p style="color: #F3E5F5; margin: 0;">Today's Registrations</p>
            </div>
            """.format(today_count), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two column layout for charts and recent activity
        chart_col, activity_col = st.columns([1.5, 1])
        
        with chart_col:
            st.markdown("### 📈 Registration Activity")
            if len(ownership_df) > 0:
                # Create a simple bar chart of registrations by date
                ownership_df['Date'] = pd.to_datetime(ownership_df['Timestamp']).dt.date
                daily_counts = ownership_df.groupby('Date').size().reset_index(name='Count')
                daily_counts = daily_counts.tail(7)  # Last 7 days
                st.bar_chart(daily_counts.set_index('Date')['Count'])
            else:
                st.info("No data to display yet.")
        
        with activity_col:
            st.markdown("### 🕐 Recent Registrations")
            if len(ownership_df) > 0:
                recent_df = ownership_df.sort_values('Timestamp', ascending=False).head(5)
                for idx, row in recent_df.iterrows():
                    st.markdown(f"""
                    <div style="background: #2a2d2a; padding: 10px 15px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #4CAF50;">
                        <strong style="color: #4CAF50;">{row['Title'][:30]}{'...' if len(row['Title']) > 30 else ''}</strong><br>
                        <small style="color: #888;">by {row['Owner']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent registrations.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # System info in a nice card
        st.markdown("### ⚙️ System Information")
        sys_col1, sys_col2 = st.columns(2)
        
        with sys_col1:
            st.markdown("""
            <div style="background: #2a2d2a; padding: 20px; border-radius: 10px;">
                <p><strong>🤖 Model:</strong> MiniLM (all-MiniLM-L6-v2)</p>
                <p><strong>📐 Algorithm:</strong> Cosine Similarity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with sys_col2:
            st.markdown("""
            <div style="background: #2a2d2a; padding: 20px; border-radius: 10px;">
                <p><strong>🔗 Storage:</strong> CSV-based simulation</p>
                <p><strong>🎯 Default Threshold:</strong> 85%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No data available yet. Start by submitting content!")