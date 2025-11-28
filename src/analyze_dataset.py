"""
Analyze dataset to find ambiguous words that appear in multiple categories
"""

import pandas as pd
from collections import defaultdict, Counter
from prepare_data import prepare_dataset


def analyze_word_distribution(df):
    """
    Analyze which words appear in multiple categories
    to identify ambiguous/non-informative words
    """
    print("=" * 80)
    print("DATASET WORD DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Create word-to-categories mapping
    word_categories = defaultdict(set)

    for idx, row in df.iterrows():
        activity = row["activity"].lower()
        category = row["category"]

        # Split into words
        words = activity.split()

        for word in words:
            # Clean word
            word = word.strip(".,!?")
            if len(word) > 2:  # Ignore very short words
                word_categories[word].add(category)

    # Find words that appear in many categories (ambiguous words)
    print("\n📊 MOST AMBIGUOUS WORDS (appear in 5+ categories):")
    print("-" * 80)

    ambiguous_words = []
    for word, categories in word_categories.items():
        if len(categories) >= 5:
            ambiguous_words.append((word, len(categories), categories))

    # Sort by number of categories
    ambiguous_words.sort(key=lambda x: x[1], reverse=True)

    for word, num_cats, categories in ambiguous_words[:20]:
        print(
            f"  '{word}' appears in {num_cats} categories: {', '.join(sorted(categories))}"
        )

    # Find category-specific words (appear in only 1 category)
    print("\n\n✅ MOST DISTINCTIVE WORDS (per category):")
    print("-" * 80)

    category_specific_words = defaultdict(list)
    for word, categories in word_categories.items():
        if len(categories) == 1:
            category = list(categories)[0]
            category_specific_words[category].append(word)

    for category in sorted(category_specific_words.keys()):
        words = category_specific_words[category]
        word_counts = Counter()

        # Count frequency of each word in this category
        for idx, row in df[df["category"] == category].iterrows():
            for word in row["activity"].lower().split():
                word = word.strip(".,!?")
                if word in words:
                    word_counts[word] += 1

        # Show top 10 most frequent distinctive words
        top_words = word_counts.most_common(10)
        if top_words:
            print(f"\n  {category.upper()}:")
            print(
                f"    {', '.join([f'{word} ({count})' for word, count in top_words])}"
            )

    # Analyze action words (beli, bayar, isi, etc.)
    print("\n\n⚠️  ACTION WORDS DISTRIBUTION:")
    print("-" * 80)

    action_words = ["beli", "bayar", "isi", "top", "ke", "di", "untuk"]

    for action in action_words:
        categories_with_action = set()
        count = 0

        for idx, row in df.iterrows():
            if action in row["activity"].lower():
                categories_with_action.add(row["category"])
                count += 1

        print(
            f"  '{action}': appears {count} times in {len(categories_with_action)} categories"
        )
        print(f"    Categories: {', '.join(sorted(categories_with_action))}")

    return ambiguous_words, category_specific_words


def suggest_stopwords(ambiguous_words, threshold=7):
    """
    Suggest words to add to stopwords based on ambiguity
    """
    print("\n\n💡 SUGGESTED STOPWORDS:")
    print("-" * 80)
    print(f"Words that appear in {threshold}+ categories should be stopwords:")
    print()

    suggested = []
    for word, num_cats, categories in ambiguous_words:
        if num_cats >= threshold:
            suggested.append(word)

    print(f"  {suggested}")
    print(f"\n  Total: {len(suggested)} words")

    return suggested


def analyze_category_confusion(df):
    """
    Find which categories are most likely to be confused
    based on word overlap
    """
    print("\n\n🔄 POTENTIAL CATEGORY CONFUSION:")
    print("-" * 80)

    # Calculate word overlap between categories
    category_words = defaultdict(set)

    for idx, row in df.iterrows():
        activity = row["activity"].lower()
        category = row["category"]

        words = set(activity.split())
        category_words[category].update(words)

    # Calculate Jaccard similarity between categories
    categories = list(category_words.keys())
    confusions = []

    for i, cat1 in enumerate(categories):
        for cat2 in categories[i + 1 :]:
            words1 = category_words[cat1]
            words2 = category_words[cat2]

            overlap = len(words1 & words2)
            union = len(words1 | words2)
            similarity = overlap / union if union > 0 else 0

            if similarity > 0.3:  # High overlap
                confusions.append((cat1, cat2, similarity, len(words1 & words2)))

    # Sort by similarity
    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\n  Category pairs with high word overlap (>30%):")
    for cat1, cat2, similarity, overlap in confusions[:10]:
        print(
            f"    {cat1} ↔️ {cat2}: {similarity:.1%} similarity ({overlap} shared words)"
        )


def main():
    print("Loading dataset...")
    df, X, y = prepare_dataset()

    print(f"\nDataset size: {len(df)} samples")
    print(f"Categories: {df['category'].nunique()}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().sort_index())

    # Run analyses
    ambiguous_words, category_specific = analyze_word_distribution(df)
    suggested_stopwords = suggest_stopwords(ambiguous_words, threshold=7)
    analyze_category_confusion(df)

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Add suggested stopwords to your model")
    print("  2. Focus on distinctive words for each category")
    print("  3. Add more training data for confused categories")
    print("  4. Use rule-based override for action+object combinations")


if __name__ == "__main__":
    main()
