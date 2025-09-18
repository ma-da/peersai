import os, json, time, hashlib, contextlib
from pathlib import Path
import requests
from config import *
from utils import *


def pin_one_file(path: Path, name_in_ipfs: str) -> str:
    """
    Pins a single file to IPFS via Pinata and returns the CID (IpfsHash).
    name_in_ipfs is used as the Pinata name/metadata only; it does NOT create folders.
    """
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    # Each request includes exactly one 'file' part -> avoids the "More than one file..." 400
    files = {
        "file": (name_in_ipfs, open(path, "rb"), "application/octet-stream")
    }
    data = {
        "pinataMetadata": json.dumps({"name": name_in_ipfs}),
        # No wrapWithDirectory here; pinning individual files
    }
    headers = {**HEADERS_AUTH}

    # Retry loop
    for attempt in range(1, IPFS_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=600)
            if not resp.ok:
                # helpful error body
                print(f"[{name_in_ipfs}] Status:", resp.status_code, "Body:", resp.text)
                resp.raise_for_status()
            return resp.json()["IpfsHash"]
        except Exception as e:
            if attempt == IPFS_MAX_RETRIES:
                raise
            time.sleep(IPFS_BACKOFF_SEC * attempt)
        finally:
            with contextlib.suppress(Exception):
                files["file"][1].close()

def upload_files_to_ipfs():
    assert IPFS_FOLDER.is_dir(), IPFS_FOLDER

    # Build per-file manifest
    entries = []
    for root, _, files in os.walk(IPFS_FOLDER):
        for fname in files:
            p = Path(root) / fname
            rel = p.relative_to(IPFS_FOLDER).as_posix()

            cid = pin_one_file(p, name_in_ipfs=rel)
            entry = {
                "name": rel,
                "size": p.stat().st_size,
                "sha256": sha256_file(p),
                "cid": cid,
                "url_dedicated": IPFS_DEDICATED + cid,
                "url_custom":    IPFS_CUSTOM + cid
            }
            entries.append(entry)
            print(f"Pinned {rel} -> {cid}")

    # Save the by-file manifest locally
    manifest = {
        "version": 1,
        "pinned_at": int(time.time()),
        "files": sorted(entries, key=lambda x: x["name"]),
        "gateways": {"dedicated": IPFS_DEDICATED, "custom": IPFS_CUSTOM}
    }
    MANIFEST_PATH = IPFS_FOLDER / "betaslim_manifest.byfile.json"
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Wrote", MANIFEST_PATH)

    # Convenience file: map group filename -> full custom-gateway URL by CID
    groups_urls = {
        e["name"]: e["url_custom"]
        for e in entries
        if e["name"].startswith("group_") and e["name"].endswith(".json")
    }
    with open(IPFS_FOLDER / "groups_urls.json", "w", encoding="utf-8") as f:
        json.dump(groups_urls, f, ensure_ascii=False, indent=2)
    print("Wrote", IPFS_FOLDER / "groups_urls.json")

    results = {}
    for path in IPFS_FILES_TO_PIN:
        files = {"file": (path.name, open(path, "rb"), "application/json")}
        try:
            resp = requests.post(
                "https://api.pinata.cloud/pinning/pinFileToIPFS",
                headers=HEADERS_AUTH,
                files=files,
                data={"pinataMetadata": json.dumps({"name": path.name})},
                timeout=600,
            )
            resp.raise_for_status()
            cid = resp.json()["IpfsHash"]
            results[path.name] = {
                "cid": cid,
                "url_dedicated": f"https://peers.mypinata.cloud/ipfs/{cid}",
                "url_custom": f"https://ai.peerservice.org/ipfs/{cid}",
            }
            print(f"Pinned {path.name} -> {cid}")
        finally:
            with contextlib.suppress(Exception):
                files["file"][1].close()

    print("\nSummary:")
    print(json.dumps(results, indent=2))

# N - looks like retreive not store. I can add storage code but have been storing it manually until now
def store_manifest_group_urls_on_hive():
    c = rpc(HIVE_RPC, "condenser_api.get_content", [HIVE_AUTHOR, HIVE_PERMLINK])

    print("Author:", c["author"])
    print("Permlink:", c["permlink"])
    print("Created (UTC):", c["created"])
    print("Is root post?:", c["parent_author"] == "")

    print("\n--- BODY ---\n")
    print(c["body"])
