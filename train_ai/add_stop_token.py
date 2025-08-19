import argparse

def add_stop_tokens(input_file, output_file, stop_token="</s>"):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if line:  # skip empty lines
                outfile.write(f"{line} {stop_token}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append stop tokens after each article line."
    )
    parser.add_argument("input_file", help="Path to input TXT file")
    parser.add_argument("output_file", help="Path to output TXT file")
    parser.add_argument("--stop_token", default="</s>", help="Stop token to append (default: </s>)")

    args = parser.parse_args()
    add_stop_tokens(args.input_file, args.output_file, args.stop_token)
    print(f"✅ Done! Wrote processed file to {args.output_file}")

