from config import *
from utils import *
import tqdm

# Take a sample inclusion list and generate an output list using a ruleset
def generate_inclusion_list(input_list, output_list_file):
    # Paths to your files
    file1 = str(BASE / "corpus_file_list_by_size.txt")
    file2 = str(BASE / "datasources/blackvault_corpus_file_list_by_size.txt")

    # Load as dataframes
    df1 = pd.read_csv(file1, sep=" ", names=["size", "filename"], engine="python")
    df2 = pd.read_csv(file2, sep=" ", names=["size", "filename"], engine="python")

    # Sort by size (largest first)
    df1_sorted = df1.sort_values(by="size", ascending=False)
    df2_sorted = df2.sort_values(by="size", ascending=False)

    # Remove smallest files

    df1 = df1[df1["size"] >= 650].copy()
    df2 = df2[df2["size"] >= 650].copy()

    # Define substrings to remove
    heavy_patterns = ["s3.documentcloud", "files.usrtk", "wp-content_uploads"]

    # Build regex pattern (joined with | means "OR")
    pattern = "|".join(heavy_patterns)

    # Filter both dataframes
    df1 = df1[~df1["filename"].str.contains(pattern, na=False)].copy()
    df2 = df2[~df2["filename"].str.contains(pattern, na=False)].copy()

    # Cut pdfs from black vault
    df2 = df2[~df2["filename"].str.contains("pdf", na=False)].copy()

    total_size_df1 = df1["size"].sum()
    total_size_df2 = df2["size"].sum()

    print("Total size of df1:", total_size_df1)
    print("Total size of df2:", total_size_df2)

    df1_sorted = df1.sort_values(by="size", ascending=False)
    corpus_main_list_filename = str(BASE / "corpus_main_list.txt")
    df1_sorted.to_csv(corpus_main_list_filename, sep=" ", index=False, header=False)

    # Sort df2 and save
    df2_sorted = df2.sort_values(by="size", ascending=False)
    corpus_blackvault_list_filename = str(BASE / "corpus_blackvault_list.txt")
    df2_sorted.to_csv(corpus_blackvault_list_filename, sep=" ", index=False, header=False)

    dfscrub = pd.read_csv(corpus_main_list_filename, sep=" ", names=["size", "filename"], engine="python")
    total_size_scrub = dfscrub["size"].sum()

    print("Total size of df1:", total_size_scrub)


def clean_corpus(files):
    stats = {
        "total_files": len(files),
        "clean_kept": 0,
        "rejected": 0,
    }

    log_rows = []
    pbar = tqdm(files, desc="Cleaning files")
    for src in pbar:
        try:
            raw = read_text_with_detection(src)
            stripped = strip_html(raw, src.suffix.lower())
            cleaned = normalize_and_clean(stripped)

            ok_run = has_natural_language_run(cleaned)
            junky = is_garbled(cleaned)

            decision = "keep" if (ok_run and not junky) else "reject"

            if decision == "keep":
                dst = relative_output_path(src)
                if not DRY_RUN:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w", encoding="utf-8", newline="\n") as f:
                        f.write(cleaned)
                stats["clean_kept"] += 1
            else:
                # Move original to rejected/ (keep relative path)
                rej = REJECT_DIR / src.relative_to(INPUT_DIR)
                if not DRY_RUN:
                    rej.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(rej))
                stats["rejected"] += 1

            log_rows.append({
                "source_path": str(src),
                "decision": decision,
                "clean_chars": len(cleaned),
                "total_words": len(WORD_RE.findall(cleaned)),
            })

        except Exception as e:
            log_rows.append({
                "source_path": str(src),
                "decision": f"error: {type(e).__name__}: {e}",
                "clean_chars": 0,
                "total_words": 0,
            })

    # Write log
    if not DRY_RUN:
        LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source_path", "decision", "clean_chars", "total_words"])
            w.writeheader()
            w.writerows(log_rows)

    print("DONE\n", stats)
    print(f"Log at: {LOG_CSV}")
    print(f"Rejected originals moved under: {REJECT_DIR}")
    print(f"Cleaned texts under: {FILTERED_CORPUS_OUTPUT_DIR}")


