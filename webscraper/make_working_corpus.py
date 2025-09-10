"""
This script assists in building up a working corpus using a subset of a full corpus.
You pass to it three arguments:
 - text file with list of files to includes from the full corpus
 - directory of the full corpus
 - target directory to output to
"""
import sys
import os
import shutil

def main():
    if len(sys.argv) != 4:
        print("make_working_corpus.py <text_file_inclusion_list> <src_corpus_dir> <output_dir>")
        return

    inclusion_list_file = sys.argv[1]
    src_dir = sys.argv[2]
    target_dir = sys.argv[3]
    print(f"Making working corpus from list '{inclusion_list_file}' with src dir '{src_dir}' and target dir '{target_dir}'")

    if len(src_dir) == 0:
        print("Error: Source directory was empty")
        return

    if len(target_dir) == 0:
        print("Error: Source directory was empty")
        return

    if src_dir == target_dir:
        print("Error: Source directory cannot equal target directory")
        return

    if not os.path.isdir(src_dir):
        print(f"Error: Source corpus directory does not exist: {src_dir}")
        return

    os.makedirs(target_dir, 0o777, exist_ok=True)

    if src_dir[len(src_dir) - 1] != '/':
        src_dir = src_dir + "/"
    if target_dir[len(target_dir) - 1] != '/':
        target_dir = target_dir + "/"

    inclusion_list = []
    try:
        with open(inclusion_list_file, 'r') as file:
            for line in file:
                inclusion_list.append(line.strip())
    except FileNotFoundError:
        print(f"Error: Inclusion list {inclusion_list_file} was not found")
        return
    except IOError:
        print(f"Error: IOError when opening inclusion list {inclusion_list_file}")
        return
    except Exception as e:
        print(f"Error: exception occurred: {e}")
        return

    print(f"Loaded inclusion list with size {len(inclusion_list)}")

    num_errors = 0
    max_allowed_errors = 10
    for file_to_include in inclusion_list:
        if len(file_to_include) <= 2:
            print(f"Skip file {file_to_include}")
            continue
        try:
            source_file = src_dir + file_to_include
            target_file = target_dir + file_to_include
            print(f"Copying file src '{source_file}' to dest '{target_file}'", flush=True)
            shutil.copy(source_file, target_file)
        except FileNotFoundError:
            print(f"Error: File to copy {file_to_include} was not found")
        except IOError as e:
            print(f"Error: IOError when trying to copy file {file_to_include}: {e}")
        except Exception as e:
            print(f"Error: exception occurred when trying to copy file: {e}")

    print ("\n ** Made working corpus **\n")
if __name__ == "__main__":
    main()