import pandas as pd
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

def decode_readable_tokens(tokenizer, string_tokens):
    """
    Convert tokens to human-readable form using the tokenizer's own decode
    logic, grouped into contiguous runs at word boundaries.

    IMPORTANT: this must decode tokens in GROUPS, not one-by-one. Some
    tokenizers (e.g. Llama-3's standard vocab on Amharic) fall back to
    single UTF-8 BYTES per token when a character has no dedicated merge.
    Since Ge'ez characters are 3 bytes in UTF-8, a lone byte token decoded
    in isolation is not valid UTF-8 by itself and will render as the
    replacement character '�'. Decoding a run of tokens together lets the
    bytes recombine into valid characters before decoding.

    Returns a list the SAME LENGTH as string_tokens, where each entry is
    the readable string for the run that token belongs to (so word-boundary
    tokens carry the full decoded chunk, and continuation tokens are shown
    as empty strings to avoid repeating the same text).
    """
    # Identify word-boundary markers used by common tokenizer families
    def is_word_start(tok):
        return tok.startswith("Ġ") or tok.startswith("▁") or tok == string_tokens[0]

    # Split into runs (each run = one "word" worth of tokens)
    runs = []
    current_run = []
    for tok in string_tokens:
        if is_word_start(tok) and current_run:
            runs.append(current_run)
            current_run = [tok]
        else:
            current_run.append(tok)
    if current_run:
        runs.append(current_run)

    # Decode each run as a whole, then map back to per-token output
    readable = []
    for run in runs:
        decoded_run = tokenizer.convert_tokens_to_string(run)
        readable.append(decoded_run)          # full decoded chunk on first token of run
        readable.extend([""] * (len(run) - 1))  # continuation tokens shown blank

    return readable


def decode_readable_string(tokenizer, string_tokens):
    """Convenience: full readable sentence (all runs decoded and joined)."""
    return tokenizer.convert_tokens_to_string(string_tokens)


def analyze_tokenization_effects(text_samples, model_path, model_label, language_label):
    """Per-sentence tokenization metrics for one model, tagged by language."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    results = []
    for pair_id, text in enumerate(text_samples, start=1):
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        fertility = token_count / word_count if word_count > 0 else 0
        cpt = char_count / token_count if token_count > 0 else 0

        string_tokens = tokenizer.convert_ids_to_tokens(tokens)
        readable_tokens = decode_readable_tokens(tokenizer, string_tokens)
        readable_sentence = decode_readable_string(tokenizer, string_tokens)

        results.append({
            "Pair ID": pair_id,
            "Language": language_label,
            "Model/Tokenizer": model_label,
            "Text": text,
            "Word Count": word_count,
            "Token Count": token_count,
            "Fertility Ratio": fertility,
            "Chars Per Token (CPT)": cpt,
            "All Tokens": string_tokens,                    # raw tokenizer output (may be byte-remapped)
            "All Tokens (Readable)": readable_tokens,        # decoded per word-run; continuation tokens blank
            "Readable Sentence": readable_sentence,          # full sentence reconstructed (sanity check)
            "Fragmented Tokens Example": string_tokens[:8],  # short raw preview
            "Fragmented Tokens Example (Readable)": readable_tokens[:8],  # short readable preview
        })

    return pd.DataFrame(results)


def summarize_with_ci(df, model_label, language_label, confidence=0.95):
    """Aggregate per-sentence metrics into mean + CI per model, per language."""
    summary = {}
    for metric in ["Fertility Ratio", "Chars Per Token (CPT)"]:
        values = df[metric].values
        n = len(values)
        mean = np.mean(values)
        sem = stats.sem(values) if n > 1 else 0.0
        margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0.0
        summary[f"{metric} (mean)"] = round(mean, 3)
        summary[f"{metric} (95% CI low)"] = round(mean - margin, 3)
        summary[f"{metric} (95% CI high)"] = round(mean + margin, 3)
    summary["Model/Tokenizer"] = model_label
    summary["Language"] = language_label
    summary["N sentences"] = len(df)
    return summary


# --- Parallel Amharic <-> English sentence pairs ---
# Each pair covers the same content/domain so cross-language fertility is
# comparable (same topics: civic life, agriculture, health, infrastructure).
# Expand toward 20-30+ pairs for a stronger CI in the final paper.
parallel_pairs = [
    {
        "am": "በማህረሰባችን ውስጥ በተሳሳተ መረጃ ምክንያት የሚፈጠረውን ችግር መከላከል አለብን።",
        "en": "We must prevent the problems caused by misinformation in our community.",
    },
    {
        "am": "የኢትዮጵያ ግብርና ሚኒስቴር በድርቅ ክፍለ ጊዜ ገበሬዎችን ለመደገፍ እቅድ አውጥቷል።",
        "en": "Ethiopia's Ministry of Agriculture has developed a plan to support farmers during drought periods.",
    },
    {
        "am": "የመንገድ ትራፊክ አደጋዎችን ለመቀነስ አዲስ የደህንነት ደንብ ወጥቷል።",
        "en": "A new safety regulation has been issued to reduce road traffic accidents.",
    },
    {
        "am": "እናቶች በእርግዝና ወቅት መደበኛ የጤና ክትትል ማድረግ አለባቸው።",
        "en": "Mothers should receive regular health checkups during pregnancy.",
    },
    {
        "am": "የአየር ንብረት ለውጥ በኢትዮጵያ የግብርና ምርታማነት ላይ ተጽዕኖ እያሳደረ ነው።",
        "en": "Climate change is affecting agricultural productivity in Ethiopia.",
    },
    {
        "am": "ተማሪዎች በትምህርት ቤት ውስጥ ጥራት ያለው ትምህርት የማግኘት መብት አላቸው።",
        "en": "Students have the right to receive quality education in schools.",
    },
    {
        "am": "የከተማው አስተዳደር አዲስ የውሃ አቅርቦት ፕሮጀክት ጀምሯል።",
        "en": "The city administration has launched a new water supply project.",
    },
    {
        "am": "በአገሪቱ የተለያዩ ክፍሎች የኢንተርኔት አገልግሎት እየተስፋፋ ነው።",
        "en": "Internet service is expanding in various parts of the country.",
    },
    {
        "am": "ወጣቶች በስራ ፈጠራ ዘርፍ ስልጠና እንዲያገኙ ይበረታታሉ።",
        "en": "Young people are encouraged to receive training in entrepreneurship.",
    },
    {
        "am": "የጤና ባለሙያዎች ወቅታዊ ክትባት አስፈላጊነትን አጽንኦት ሰጥተዋል።",
        "en": "Health professionals have emphasized the importance of timely vaccination.",
    },
]

amharic_texts = [p["am"] for p in parallel_pairs]
english_texts = [p["en"] for p in parallel_pairs]

models_to_test = [
    ("meta-llama/Meta-Llama-3-8B", "Llama-3 (Standard)"),
    ("intfloat/multilingual-e5-large", "Multilingual-E5 (Optimized)"),
    ("rasyosef/Llama-3.2-1B-Amharic-Instruct", "Llama-3.2-1B-Amharic (Specialized)"),
    ("israel/LLAMA-Walia-II", "LLAMA-Walia-II (Specialized)"),
    ("rasyosef/roberta-amharic-text-embedding-base", "RoBERTa-Amharic-Embed (Specialized)"),
    ("CohereLabs/tiny-aya-earth", "Tiny Aya Earth (Multilingual/African-focused)"),
]

per_sentence_results = []
summary_rows = []

for model_path, model_label in models_to_test:
    df_am = analyze_tokenization_effects(amharic_texts, model_path, model_label, "Amharic")
    df_en = analyze_tokenization_effects(english_texts, model_path, model_label, "English")
    per_sentence_results.extend([df_am, df_en])
    summary_rows.append(summarize_with_ci(df_am, model_label, "Amharic"))
    summary_rows.append(summarize_with_ci(df_en, model_label, "English"))

# Full per-sentence table, tagged by language and pair ID so Amharic/English
# rows for the same underlying sentence can be joined on (Model, Pair ID)
per_sentence_df = pd.concat(per_sentence_results, ignore_index=True)

# Aggregated summary table with 95% CIs, split by language — this is the
# one for the paper
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[[
    "Model/Tokenizer", "Language", "N sentences",
    "Fertility Ratio (mean)", "Fertility Ratio (95% CI low)", "Fertility Ratio (95% CI high)",
    "Chars Per Token (CPT) (mean)", "Chars Per Token (CPT) (95% CI low)", "Chars Per Token (CPT) (95% CI high)",
]]

print(summary_df.to_string(index=False))

# --- Amharic vs English "fertility gap" per model ---
# How many more tokens/word the model needs for Amharic vs English on the
# SAME underlying content. This is the headline number for the paper.
pivot = summary_df.pivot(index="Model/Tokenizer", columns="Language", values="Fertility Ratio (mean)")
pivot["Fertility Gap (Amharic - English)"] = pivot["Amharic"] - pivot["English"]
pivot["Fertility Ratio (Amharic / English)"] = pivot["Amharic"] / pivot["English"]
print("\nAmharic vs English fertility gap by model:\n")
print(pivot.round(3).to_string())

# View the full readable tokenized output for the first N pairs, across all
# models, side by side (Amharic vs English):
N = 5
print(f"\nFull tokenization (readable) for first {N} parallel pairs, by model:\n")
for pair_id in range(1, N + 1):
    pair_rows = per_sentence_df[per_sentence_df["Pair ID"] == pair_id]
    am_text = pair_rows[pair_rows["Language"] == "Amharic"]["Text"].iloc[0]
    en_text = pair_rows[pair_rows["Language"] == "English"]["Text"].iloc[0]
    print(f"Pair {pair_id}")
    print(f"  AM: {am_text}")
    print(f"  EN: {en_text}")
    print("-" * 80)
    for _, row in pair_rows.iterrows():
        print(f"  [{row['Language']}] {row['Model/Tokenizer']} "
              f"({row['Token Count']} tokens, fertility={row['Fertility Ratio']:.2f}):")
        print("    raw:      ", row["All Tokens"])
        print("    readable: ", row["All Tokens (Readable)"])
    print()

import matplotlib.pyplot as plt

# Grouped bar chart: Amharic vs English fertility, per model, with 95% CI
fig, ax = plt.subplots(figsize=(11, 5.5))

models = summary_df["Model/Tokenizer"].unique()
x = np.arange(len(models))
width = 0.35

for i, lang in enumerate(["Amharic", "English"]):
    lang_df = summary_df[summary_df["Language"] == lang].set_index("Model/Tokenizer").loc[models]
    means = lang_df["Fertility Ratio (mean)"].values
    lower_err = means - lang_df["Fertility Ratio (95% CI low)"].values
    upper_err = lang_df["Fertility Ratio (95% CI high)"].values - means
    offset = (i - 0.5) * width
    ax.bar(x + offset, means, width, yerr=[lower_err, upper_err], capsize=4,
           label=lang, color="#4C72B0" if lang == "Amharic" else "#DD8452")

ax.set_ylabel("Fertility Ratio (tokens/word)")
ax.set_title("Amharic vs English Tokenization Fertility, Parallel Sentences (95% CI)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.legend(title="Language")
plt.tight_layout()
plt.savefig("fertility_comparison_amharic_vs_english.png", dpi=150)
plt.show()

# --- Example output (parallel run, N=10 pairs, Amharic + English) ---
# Real summary_df output from a full run of the 10 parallel pairs above,
# kept here as a reference so the numbers below can be sanity-checked
# against future reruns (e.g. after expanding to 20-30+ pairs).
#
#                                Model/Tokenizer Language  N sentences  Fertility Ratio (mean)  Fertility Ratio (95% CI low)  Fertility Ratio (95% CI high)  Chars Per Token (CPT) (mean)  Chars Per Token (CPT) (95% CI low)  Chars Per Token (CPT) (95% CI high)
#                         Llama-3-8B (Standard)  Amharic           10                  13.524                         12.656                          14.392                          0.402                                0.396                                 0.408
#                         Llama-3-8B (Standard)  English           10                   1.148                          1.077                           1.219                          6.160                                5.637                                 6.683
#                   Multilingual-E5 (Optimized)  Amharic           10                   1.951                          1.709                           2.193                          2.866                                2.441                                 3.291
#                   Multilingual-E5 (Optimized)  English           10                   1.399                          1.252                           1.546                          5.068                                4.836                                 5.299
#            Llama-3.2-1B-Amharic (Specialized)  Amharic           10                   1.375                          1.284                           1.465                          3.967                                3.717                                 4.218
#            Llama-3.2-1B-Amharic (Specialized)  English           10                   1.148                          1.077                           1.219                          6.160                                5.637                                 6.683
#                  LLAMA-Walia-II (Specialized)  Amharic           10                   1.200                          1.074                           1.326                          4.599                                4.103                                 5.095
#                  LLAMA-Walia-II (Specialized)  English           10                   1.435                          1.235                           1.635                          4.986                                4.657                                 5.315
#           RoBERTa-Amharic-Embed (Specialized)  Amharic           10                   1.199                          1.060                           1.339                          4.617                                4.108                                 5.125
#           RoBERTa-Amharic-Embed (Specialized)  English           10                   4.733                          4.313                           5.153                          1.494                                1.428                                 1.560
# Tiny Aya Earth (Multilingual/African-focused)  Amharic           10                   3.291                          2.907                           3.676                          1.679                                1.520                                 1.838
# Tiny Aya Earth (Multilingual/African-focused)  English           10                   1.148                          1.077                           1.219                          6.160                                5.637                                 6.683
#
# Amharic vs English fertility gap by model:
#
# Model/Tokenizer                                Amharic  English  Fertility Gap (Amharic - English)  Fertility Ratio (Amharic / English)
# LLAMA-Walia-II (Specialized)                     1.200    1.435                             -0.235                                0.836
# Llama-3-8B (Standard)                           13.524    1.148                             12.376                               11.780
# Llama-3.2-1B-Amharic (Specialized)               1.375    1.148                              0.227                                1.198
# Multilingual-E5 (Optimized)                      1.951    1.399                              0.552                                1.395
# RoBERTa-Amharic-Embed (Specialized)              1.199    4.733                             -3.534                                0.253
# Tiny Aya Earth (Multilingual/African-focused)    3.291    1.148                              2.143                                2.867
#
# Takeaways:
# 1. Llama-3-8B (standard vocab) is the clearest failure case: fertility
#    ~13.5 tokens/word on Amharic vs ~1.1 on English (11.8x gap), with CPT
#    collapsing to ~0.4 -- it is spending MORE than one token per UTF-8
#    byte of Ge'ez text, i.e. falling back to raw byte tokens.
# 2. Two Amharic-specialized tokenizers (LLAMA-Walia-II, RoBERTa-Amharic-
#    Embed) invert the expected pattern: they tokenize Amharic MORE
#    efficiently than English (fertility ratio 0.84x and 0.25x). This is
#    a genuinely interesting/reportable anomaly, not just a sanity check
#    -- it suggests these vocabs were built with Amharic (not English) as
#    the primary target, at English's expense. Worth flagging explicitly
#    in the writeup rather than treating "specialized = better at both
#    languages" as a given.
# 3. Tiny Aya Earth and Multilingual-E5 sit in between: real fragmentation
#    on Amharic (fertility 3.3 and 1.95) but not full byte-fallback
#    collapse, consistent with partial subword coverage of Ge'ez script.
# 4. Llama-3.2-1B-Amharic is the only specialized model that still favors
#    English fertility (1.148) over Amharic (1.375) while keeping both
#    reasonably low (gap ratio 1.2x) -- arguably the best-balanced
#    tokenizer of the six for a bilingual AM/EN use case.
