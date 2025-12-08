"""
Real System Evaluation for Digital Ownership Protection System
================================================================
This script tests the plagiarism detection system using PAN-PC-11 corpus.

Usage:
  python evaluate_system.py           # Register 150 docs + test 20 samples
  python evaluate_system.py --test    # Test only (no new registrations)
"""

import os
import random
import pandas as pd
import hashlib
import re
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Check for --test flag
TEST_ONLY = "--test" in sys.argv or "-t" in sys.argv

# Configuration
PAN_BASE_PATH = "pan-pc-data/pan-plagiarism-corpus-2011"
SOURCE_DOCS_PATH = os.path.join(PAN_BASE_PATH, "external-detection-corpus", "source-document")
NUM_REGISTER = 150  # Documents to register (100-200 range)
NUM_TESTS = 20      # Test samples
THRESHOLD = 0.85    # Same as app.py default
CHUNK_SIZE = 800    # Characters per chunk

# Use SAME file paths as app.py
OWNERSHIP_FILE = 'ownership_records.csv'
CONTENT_FILE = 'registered_content.csv'

print("=" * 60)
print("REAL SYSTEM EVALUATION")
if TEST_ONLY:
    print("MODE: Test Only (no new registrations)")
else:
    print("MODE: Full (register + test)")
print("=" * 60)
print()

# Initialize model (same as app.py)
print("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[OK] Model loaded\n")

# ===== SAME FUNCTIONS AS APP.PY =====

def preprocess_text(text):
    """Clean and normalize text - SAME AS APP.PY"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def compute_similarity(new_text, existing_embeddings):
    """Calculate cosine similarity - SAME AS APP.PY"""
    new_embedding = model.encode([new_text])
    if len(existing_embeddings) == 0:
        return 0.0, -1
    similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)
    return max_similarity, max_index

def check_originality(text, threshold=0.85):
    """Check if text is original - SAME AS APP.PY"""
    processed_text = preprocess_text(text)
    
    if os.path.exists(CONTENT_FILE):
        content_df = pd.read_csv(CONTENT_FILE)
        
        if len(content_df) > 0:
            existing_embeddings = []
            for idx, row in content_df.iterrows():
                embedding = model.encode([row['Content']])
                existing_embeddings.append(embedding[0])
            
            existing_embeddings = np.array(existing_embeddings)
            similarity, match_index = compute_similarity(processed_text, existing_embeddings)
            
            if similarity >= threshold:
                matched_content = content_df.iloc[match_index]
                return False, similarity, matched_content
            else:
                return True, similarity, None
            
    return True, 0.0, None

def register_content(text, owner, title):
    """Register content to database - SAME AS APP.PY"""
    processed = preprocess_text(text)
    embedding = model.encode([processed])[0].tolist()
    timestamp = datetime.now().isoformat()
    
    unique_str = processed + owner + timestamp
    content_hash = hashlib.sha256(unique_str.encode()).hexdigest()
    nft_hash = hashlib.sha256((content_hash + owner).encode()).hexdigest()
    
    # Initialize files if needed
    if not os.path.exists(CONTENT_FILE):
        pd.DataFrame(columns=['Content_Hash', 'Content', 'Embedding', 'Timestamp']).to_csv(CONTENT_FILE, index=False)
    if not os.path.exists(OWNERSHIP_FILE):
        pd.DataFrame(columns=['NFT_Hash', 'Owner', 'Title', 'Timestamp', 'Content_Hash']).to_csv(OWNERSHIP_FILE, index=False)
    
    # Save to content file
    content_df = pd.read_csv(CONTENT_FILE)
    content_df = pd.concat([content_df, pd.DataFrame([{
        'Content_Hash': content_hash,
        'Content': processed,
        'Embedding': str(embedding),
        'Timestamp': timestamp
    }])], ignore_index=True)
    content_df.to_csv(CONTENT_FILE, index=False)
    
    # Save to ownership file
    ownership_df = pd.read_csv(OWNERSHIP_FILE)
    ownership_df = pd.concat([ownership_df, pd.DataFrame([{
        'NFT_Hash': nft_hash,
        'Owner': owner,
        'Title': title,
        'Timestamp': timestamp,
        'Content_Hash': content_hash
    }])], ignore_index=True)
    ownership_df.to_csv(OWNERSHIP_FILE, index=False)
    
    return processed

# ===== PAN CORPUS FUNCTIONS =====

def get_source_documents():
    """Get list of source document paths from PAN corpus"""
    source_docs = []
    for part_dir in os.listdir(SOURCE_DOCS_PATH):
        part_path = os.path.join(SOURCE_DOCS_PATH, part_dir)
        if os.path.isdir(part_path):
            for filename in os.listdir(part_path):
                if filename.endswith('.txt'):
                    source_docs.append(os.path.join(part_path, filename))
    return source_docs

def read_document(filepath, chunk_size=CHUNK_SIZE):
    """Read a meaningful chunk from a document"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            content = content[500:] if len(content) > 500 else content
            return content[:chunk_size].strip()
    except Exception:
        return None

def paraphrase_text(text):
    """Simple paraphrase by sentence shuffling"""
    sentences = text.split('. ')
    if len(sentences) > 3:
        middle = sentences[1:-1]
        random.shuffle(middle)
        return '. '.join([sentences[0]] + middle + [sentences[-1]])
    return text

# ===== MAIN EVALUATION =====

def run_evaluation():
    # Check database first
    if os.path.exists(CONTENT_FILE):
        content_df = pd.read_csv(CONTENT_FILE)
        print(f"Current database: {len(content_df)} registered documents\n")
    else:
        content_df = pd.DataFrame()
        print("Database is empty!\n")
    
    print("Finding PAN-PC source documents...")
    all_docs = get_source_documents()
    print(f"Found {len(all_docs)} source documents\n")
    
    results = {
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0
    }
    
    if TEST_ONLY:
        # ===== TEST ONLY MODE =====
        print("-" * 60)
        print("PHASE 1: Skipped (test-only mode)")
        print("-" * 60)
        
        if len(content_df) == 0:
            print("\nError: No content in database to test against!")
            print("Run without --test flag first to register samples.")
            return
        
        # Sample existing content from database
        registered_texts = content_df['Content'].tolist()
        print(f"Using {len(registered_texts)} existing documents for testing\n")
        
    else:
        # ===== FULL MODE: REGISTER + TEST =====
        if len(all_docs) < NUM_REGISTER + NUM_TESTS:
            print(f"Warning: Not enough documents! Need {NUM_REGISTER + NUM_TESTS}")
            return
        
        selected_docs = random.sample(all_docs, NUM_REGISTER + NUM_TESTS)
        register_docs = selected_docs[:NUM_REGISTER]
        
        print("-" * 60)
        print(f"PHASE 1: Registering {NUM_REGISTER} documents")
        print("-" * 60)
        
        registered_texts = []
        for i, doc_path in enumerate(register_docs):
            chunk = read_document(doc_path)
            if chunk and len(chunk) > 100:
                doc_name = os.path.basename(doc_path)
                register_content(chunk, "PAN-Corpus", f"Source-{doc_name[:20]}")
                registered_texts.append(chunk)
                if (i + 1) % 25 == 0:
                    print(f"  Registered {i+1}/{NUM_REGISTER} documents...")
        
        print(f"\n[OK] Registered {len(registered_texts)} documents to database\n")
    
    # ===== PHASE 2: Testing =====
    print("-" * 60)
    print(f"PHASE 2: Testing with {NUM_TESTS} samples")
    print("-" * 60)
    
    # Test A: Exact copies (should be REJECTED)
    print("\n[Test A] Exact copies of registered content (should REJECT)")
    test_copies = random.sample(registered_texts, min(7, len(registered_texts)))
    for i, text in enumerate(test_copies):
        is_original, sim, _ = check_originality(text, THRESHOLD)
        if not is_original:
            results['true_positive'] += 1
            status = "DETECTED"
        else:
            results['false_negative'] += 1
            status = "MISSED"
        print(f"  [{i+1}] Similarity: {sim:.2%} - {status}")
    
    # Test B: Paraphrased copies (should show high similarity)
    print("\n[Test B] Paraphrased content (should show high similarity)")
    test_paraphrased = random.sample(registered_texts, min(6, len(registered_texts)))
    for i, text in enumerate(test_paraphrased):
        paraphrased = paraphrase_text(text)
        is_original, sim, _ = check_originality(paraphrased, THRESHOLD)
        if not is_original:
            results['true_positive'] += 1
            status = "DETECTED"
        else:
            results['false_negative'] += 1
            status = "Below threshold"
        print(f"  [{i+1}] Similarity: {sim:.2%} - {status}")
    
    # Test C: Completely NEW documents (should be ACCEPTED)
    print("\n[Test C] New unregistered content (should ACCEPT)")
    test_docs = random.sample(all_docs, 20)  # Get random docs for testing
    tested = 0
    for doc_path in test_docs:
        chunk = read_document(doc_path)
        if chunk and len(chunk) > 100:
            is_original, sim, _ = check_originality(chunk, THRESHOLD)
            if is_original:
                results['true_negative'] += 1
                status = "ACCEPTED"
            else:
                results['false_positive'] += 1
                status = "FALSE ALARM"
            tested += 1
            print(f"  [{tested}] Similarity: {sim:.2%} - {status}")
            if tested >= 7:
                break
    
    # ===== Calculate Metrics =====
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    tp = results['true_positive']
    fp = results['false_positive']
    tn = results['true_negative']
    fn = results['false_negative']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp} (correctly detected plagiarism)")
    print(f"  True Negatives:  {tn} (correctly accepted original)")
    print(f"  False Positives: {fp} (wrongly flagged as plagiarism)")
    print(f"  False Negatives: {fn} (missed actual plagiarism)")
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"\nThreshold used: {THRESHOLD:.0%}")
    print("=" * 60)
    
    # Check current database size
    final_df = pd.read_csv(CONTENT_FILE)
    print(f"\n[INFO] Database contains {len(final_df)} registered documents")

if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
