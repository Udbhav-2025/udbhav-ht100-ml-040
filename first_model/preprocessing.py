import pandas as pd
import re
import emoji
from collections import Counter

# Statistics tracking
stats = {
    'original_rows': 0,
    'removed_empty': 0,
    'removed_emoji_only': 0,
    'removed_punctuation_only': 0,
    'removed_gibberish': 0,
    'removed_non_english': 0,
    'duplicates_removed': 0,
    'final_rows': 0
}

def is_emoji_only(text):
    """Check if text contains only emojis and whitespace"""
    text_without_emoji = emoji.replace_emoji(text, '')
    return len(text_without_emoji.strip()) == 0

def is_punctuation_only(text):
    """Check if text is only punctuation like '.', '-', etc."""
    return bool(re.match(r'^[\s\.\-\,\;\:\!\?]+$', text))

def is_gibberish(text):
    """Detect gibberish: very long words, excessive repeated chars, random patterns"""
    words = text.split()
    
    # Check for excessively long words (likely gibberish)
    if any(len(word) > 30 for word in words):
        return True
    
    # Check for too many consecutive repeated characters
    if re.search(r'(.)\1{5,}', text):
        return True
    
    # Check if text has very low vowel ratio (might be gibberish)
    vowels = len(re.findall(r'[aeiouAEIOU]', text))
    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', text))
    if consonants > 0 and vowels / (consonants + vowels) < 0.15:
        return True
    
    return False

def is_non_english(text):
    """Basic check for non-English text (has non-ASCII characters excluding emojis)"""
    # Remove emojis first
    text_no_emoji = emoji.replace_emoji(text, '')
    # Check if contains non-ASCII characters
    try:
        text_no_emoji.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True

def keep_emotional_emojis(text):
    """Keep only emotional emojis relevant to sentiment/suicide detection"""
    emotional_emojis = set([
        '❤️', '💔', '😔', '😭', '😢', '😞', '😟', '😥', '😰', '😨', 
        '😱', '😖', '😣', '😓', '😩', '😫', '🥺', '😪', '💀', '☠️',
        '🙁', '☹️', '😦', '😧', '😮', '😯', '🥀', '🖤', '💙', '💚',
        '💛', '🧡', '💜', '🤍', '🤎', '💗', '💖', '💝', '🤗', '🥰',
        '😊', '😌', '🙏', '😇'
    ])
    
    result = []
    for char in text:
        if char in emoji.EMOJI_DATA:
            if char in emotional_emojis:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)

def remove_urls_mentions_hashtags(text):
    """Remove URLs, @mentions, and #hashtags"""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    return text

def fix_repeated_chars(text):
    """Fix repeated characters (e.g., 'soooo sad' -> 'soo sad')"""
    # Allow max 2 repetitions of any character
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def normalize_text(text):
    """Normalize text: lowercase, strip whitespace, remove excessive punctuation"""
    # Lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags
    text = remove_urls_mentions_hashtags(text)
    
    # Keep only emotional emojis
    text = keep_emotional_emojis(text)
    
    # Fix repeated characters
    text = fix_repeated_chars(text)
    
    # Remove excessive punctuation (more than 2 consecutive)
    text = re.sub(r'([!?.;,]){3,}', r'\1\1', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def clean_suicide_detection_data(input_file, output_file):
    """Main function to clean the Suicide Detection CSV"""
    
    print("Loading CSV file...")
    df = pd.read_csv(input_file)
    
    stats['original_rows'] = len(df)
    print(f"\n{'='*60}")
    print(f"ORIGINAL DATA: {stats['original_rows']} rows")
    print(f"{'='*60}\n")
    
    # Identify the text column (usually 'text', 'tweet', or similar)
    text_column = None
    for col in df.columns:
        if col.lower() in ['text', 'tweet', 'post', 'message', 'content']:
            text_column = col
            break
    
    if text_column is None:
        # Use the first column that looks like text
        text_column = df.columns[0]
    
    print(f"Using column: '{text_column}' for text data\n")
    
    # Convert to string and handle NaN
    df[text_column] = df[text_column].fillna('').astype(str)
    
    # A. Remove useless rows
    print("STEP A: Removing useless rows...")
    print("-" * 60)
    
    # 1. Empty text
    mask_empty = df[text_column].str.strip() == ''
    stats['removed_empty'] = mask_empty.sum()
    print(f"  - Empty text: {stats['removed_empty']} rows")
    
    # 2. Emoji only
    mask_emoji_only = df[text_column].apply(is_emoji_only)
    stats['removed_emoji_only'] = mask_emoji_only.sum()
    print(f"  - Emoji only: {stats['removed_emoji_only']} rows")
    
    # 3. Punctuation only (".", "-", etc.)
    mask_punct_only = df[text_column].apply(is_punctuation_only)
    stats['removed_punctuation_only'] = mask_punct_only.sum()
    print(f"  - Punctuation only: {stats['removed_punctuation_only']} rows")
    
    # 4. Gibberish
    mask_gibberish = df[text_column].apply(is_gibberish)
    stats['removed_gibberish'] = mask_gibberish.sum()
    print(f"  - Gibberish: {stats['removed_gibberish']} rows")
    
    # 5. Non-English (optional - uncomment if needed)
    # mask_non_english = df[text_column].apply(is_non_english)
    # stats['removed_non_english'] = mask_non_english.sum()
    # print(f"  - Non-English: {stats['removed_non_english']} rows")
    
    # Combine all removal masks
    mask_to_remove = mask_empty | mask_emoji_only | mask_punct_only | mask_gibberish  # | mask_non_english
    df_clean = df[~mask_to_remove].copy()
    
    print(f"\nRows after Step A: {len(df_clean)}")
    print()
    
    # B. Normalize text
    print("STEP B: Normalizing text...")
    print("-" * 60)
    print("  - Converting to lowercase")
    print("  - Stripping whitespace")
    print("  - Removing excessive punctuation")
    print("  - Fixing repeated characters")
    print("  - Removing URLs, mentions, hashtags")
    print("  - Keeping only emotional emojis")
    
    df_clean[text_column] = df_clean[text_column].apply(normalize_text)
    
    # Remove any rows that became empty after normalization
    mask_empty_after = df_clean[text_column].str.strip() == ''
    empty_after_count = mask_empty_after.sum()
    if empty_after_count > 0:
        print(f"\n  - Removed {empty_after_count} rows that became empty after normalization")
        stats['removed_empty'] += empty_after_count
        df_clean = df_clean[~mask_empty_after]
    
    print(f"\nRows after Step B: {len(df_clean)}")
    print()
    
    # C. Remove duplicates
    print("STEP C: Removing duplicates...")
    print("-" * 60)
    
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[text_column], keep='first')
    stats['duplicates_removed'] = before_dedup - len(df_clean)
    
    print(f"  - Duplicates removed: {stats['duplicates_removed']} rows")
    print(f"\nRows after Step C: {len(df_clean)}")
    print()
    
    # Final statistics
    stats['final_rows'] = len(df_clean)
    
    # Save cleaned data
    print("Saving cleaned data...")
    df_clean.to_csv(output_file, index=False)
    print(f"✓ Cleaned data saved to: {output_file}\n")
    
    # Print summary
    print(f"{'='*60}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Original rows:              {stats['original_rows']:,}")
    print(f"\nRemoved:")
    print(f"  - Empty text:             {stats['removed_empty']:,}")
    print(f"  - Emoji only:             {stats['removed_emoji_only']:,}")
    print(f"  - Punctuation only:       {stats['removed_punctuation_only']:,}")
    print(f"  - Gibberish:              {stats['removed_gibberish']:,}")
    # print(f"  - Non-English:            {stats['removed_non_english']:,}")
    print(f"  - Duplicates:             {stats['duplicates_removed']:,}")
    print(f"\nFinal rows:                 {stats['final_rows']:,}")
    
    rows_removed = stats['original_rows'] - stats['final_rows']
    percentage_removed = (rows_removed / stats['original_rows']) * 100
    percentage_kept = 100 - percentage_removed
    
    print(f"\nTotal removed:              {rows_removed:,} ({percentage_removed:.2f}%)")
    print(f"Retention rate:             {percentage_kept:.2f}%")
    print(f"{'='*60}\n")
    
    # Show some examples
    print("Sample of cleaned data:")
    print("-" * 60)
    print(df_clean[text_column].head(10).to_string(index=False))
    print(f"{'='*60}\n")

if __name__ == "__main__":
    input_file = "Suicide_Detection.csv"
    output_file = "Suicide_Detection_cleaned.csv"
    
    clean_suicide_detection_data(input_file, output_file)
