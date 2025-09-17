import storage

def main():
    storage.upload_files_to_ipfs()

    storage.store_manifest_group_urls_on_hive()

if __name__ == "__main__":
    main()
