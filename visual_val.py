import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import warnings

# Suppress pandas future warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Paths to your local Gaokao data directories
DATA_DIRS = [
    "./GAOKAO-Bench-2010-2022/Data",
    "./GAOKAO-Bench-2023-2024/Data"
]

# The key used in the JSON files to represent the specific model's score.
# Adjust based on your specific JSON schema (e.g., 'score', 'gpt4_score', 'is_correct')
SCORE_KEY = "score" 

# The semantic embedding model (BAAI/bge-m3 is the current state-of-the-art for Chinese text)
EMBEDDING_MODEL_NAME = "models/bge-m3"

# Number of random question pairs to sample
NUM_PAIRS = 100_000 
# ==========================================

def load_gaokao_evaluations(data_dirs, score_key):
    """
    Recursively scans the directories for JSON files and extracts the questions and scores.
    """
    questions = []
    scores = []
    
    print("Scanning directories for evaluation files...")
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"  [!] Warning: Directory '{data_dir}' not found. Please check the path.")
            continue
            
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json') or file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            # Handle standard JSON vs JSONL
                            if file.endswith('.json'):
                                data = json.load(f)
                                # Gaokao-Bench JSONs can be a list or a dict containing an 'example'/'data' list
                                if isinstance(data, dict):
                                    items = data.get("example", data.get("data", [data]))
                                elif isinstance(data, list):
                                    items = data
                                else:
                                    continue
                            else:
                                items = [json.loads(line) for line in f if line.strip()]
                                
                            for item in items:
                                if not isinstance(item, dict):
                                    continue
                                
                                # Extract the question text
                                q_text = item.get("question") or item.get("instruction") or item.get("content")
                                
                                # Extract the correctness score
                                score = item.get(score_key)
                                if score is None and "is_correct" in item:
                                    score = item["is_correct"]
                                
                                if q_text and score is not None:
                                    try:
                                        # Standardize score to a binary integer: 1 (correct) or 0 (incorrect)
                                        binary_score = 1 if float(score) > 0.5 else 0
                                        questions.append(str(q_text))
                                        scores.append(binary_score)
                                    except (ValueError, TypeError):
                                        continue
                    except Exception as e:
                        pass # Silently skip files that cannot be parsed
                        
    return questions, scores

def main():
    # Step A: Load Data
    questions, scores = load_gaokao_evaluations(DATA_DIRS, SCORE_KEY)
    num_questions = len(questions)
    
    if num_questions < 100:
        print(f"Error: Only {num_questions} questions loaded. Please verify your DATA_DIRS and SCORE_KEY.")
        return
        
    print(f"Successfully loaded {num_questions} evaluated questions.")
    
    # Calculate Expected Statistical Baseline
    # (If a model has 80% accuracy, random chance of two queries having the same result is 0.8^2 + 0.2^2 = 68%)
    overall_accuracy = sum(scores) / num_questions
    random_baseline = (overall_accuracy ** 2) + ((1 - overall_accuracy) ** 2)
    print(f"Overall Model Accuracy: {overall_accuracy:.2%}")
    print(f"Random Agreement Baseline (By Chance): {random_baseline:.2%}")

    # Step B: Generate Semantic Embeddings
    print(f"\nLoading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Generating dense vectors for all questions (this may take a moment)...")
    # Setting normalize_embeddings=True allows us to compute cosine similarity using a fast dot product
    embeddings = embedder.encode(questions, show_progress_bar=True, normalize_embeddings=True)

    # Step C: Randomly Sample Pairs & Calculate Metrics
    print(f"\nSampling {NUM_PAIRS} random question pairs...")
    idx1 = np.random.randint(0, num_questions, NUM_PAIRS)
    idx2 = np.random.randint(0, num_questions, NUM_PAIRS)
    
    # Remove identical self-pairs (Question A matched with Question A)
    valid_mask = idx1 != idx2
    idx1, idx2 = idx1[valid_mask], idx2[valid_mask]
    
    print("Calculating pairwise cosine similarities and performance agreements...")
    # Fast vectorized cosine similarity
    similarities = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)
    
    # Performance Agreement Calculation (1 if both right or both wrong, else 0)
    scores_array = np.array(scores)
    agreements = (scores_array[idx1] == scores_array[idx2]).astype(int)

    # Step D: Binning the Similarities
    df = pd.DataFrame({'Similarity': similarities, 'Agreement': agreements})
    
    # Define semantic similarity intervals (0.40 to 1.00 in steps of 0.05)
    bins = np.arange(0.40, 1.05, 0.05)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    
    df['Sim_Bin'] = pd.cut(df['Similarity'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate average agreement rate and sample count per bin
    bin_stats = df.groupby('Sim_Bin', observed=False)['Agreement'].agg(['mean', 'count']).reset_index()
    # Filter out empty or highly noisy bins (< 25 samples)
    bin_stats = bin_stats[bin_stats['count'] > 25] 

    # Step E: Visualization
    print("\nGenerating Verification Chart...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Draw bars to visualize where the distribution of similarities falls, and a line to show the trend
    ax = sns.barplot(x='Sim_Bin', y='mean', data=bin_stats, color='skyblue', alpha=0.6)
    sns.pointplot(x='Sim_Bin', y='mean', data=bin_stats, color='crimson', markers='o', scale=1.2)
    
    # Plot the expected random baseline
    plt.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=2, 
                label=f'Random Chance Baseline ({random_baseline:.2%})')
    
    plt.title('Assumption Verification: Performance Agreement vs. Semantic Similarity', fontsize=15, pad=15)
    plt.xlabel('Cosine Similarity Intervals', fontsize=13)
    plt.ylabel('Average Agreement Rate (Both Correct / Both Wrong)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    
    # Add pair counts (n) at the bottom of each bar for transparency
    for i, row in bin_stats.iterrows():
        plt.text(i, 0.05, f"n={int(row['count'])}", ha='center', fontsize=9, color='black')

    plt.tight_layout()
    output_filename = 'semantic_agreement_verification.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nVerification complete! Chart saved locally as '{output_filename}'.")

if __name__ == "__main__":
    main()