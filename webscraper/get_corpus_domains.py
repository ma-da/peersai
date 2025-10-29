from pathlib import Path
import re
import sys

def extract_domains(directory: str) -> set:
    """
    Extract unique URL domains from filenames in the specified directory.
    Ignores files without a valid web domain in their name.

    Parameters
    ----------
    directory : str
        Path to the directory containing files named after web addresses.

    Returns
    -------
    set
        A set of unique domain names extracted from valid filenames.
    """
    # Initialize a set to store unique domains
    domains = set()

    # Convert directory to Path object
    dir_path = Path(directory)

    # Check if directory exists
    if not dir_path.is_dir():
        raise ValueError(f"Directory {directory} does not exist or is not a directory")

    # Regular expression for valid web domains
    # Matches formats like example.com, www.example.co.uk, sub.example.org
    domain_pattern = re.compile(r'^(?:[a-zA-Z0-9][a-zA-Z0-9\-]*\.)+[a-zA-Z]{2,}(?=_|$)')

    # Iterate through files in the directory
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            filename = file_path.name
            # Extract the domain part before the first underscore (or end of string)
            match = domain_pattern.match(filename)
            if match:
                domain = match.group(0)
                if domain.endswith(".txt") or domain.endswith(".html") or domain.endswith(".pdf") or domain.endswith(".epub"):
                    continue
                domains.add(domain)
            #else:
            #    print(f"Skipped {filename}: no valid web domain found")

    return domains

def main():
    if len(sys.argv) < 2:
        print("get_corpus_domains.py <corpus_dir>")
        return 1

    directory = sys.argv[1]
    #print(f"Extracting domains from directory: {directory}")

    try:
        # Extract unique domains
        unique_domains = extract_domains(directory)

        # Print unique domains to stdout
        if unique_domains:
            #print("\nUnique URL domains found:")
            for domain in sorted(unique_domains):
                print(domain)
            print(f"\nTotal unique domains: {len(unique_domains)}")
        else:
            print("No valid domains found in the specified directory.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()