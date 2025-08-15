# This will post to the Hive blockchain

import requests, struct, time, json, re, hashlib, calendar
from datetime import datetime, timezone, timedelta
from ecdsa import SigningKey, SECP256k1, util as eutil
from ecdsa import VerifyingKey

# ===== USER CONFIG =====
RPC         = "https://api.hive.blog"
ACCOUNT     = "wanttoknow"                      # account name no "@"
POSTING_WIF = "THE ACCOUNT POSTING KEY"  # posting key (WIF)

# Top-level post (no parent)
parent_author   = ""                  # EMPTY for top-level
parent_permlink = "test"              # first tag or community (e.g., "hive-167922")
title = "Testing app posting"
body  = """This is a top-level post signed with pure python. If you are reading this, the test was successful."""
tags  = ["test", "jupyter", "python"]
json_metadata = {"tags": tags, "app": "jupyter/0.1"}

# ===== CONSTANTS =====
CHAIN_ID = bytes.fromhex("beeab0de00000000000000000000000000000000000000000000000000000000")  # Hive mainnet
COMMENT_OP_ID = 1
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
N_CURVE = SECP256k1.order

# ===== HELPERS =====
def b58decode(b58s: str) -> bytes:
    n = 0
    for ch in b58s:
        n = n * 58 + BASE58_ALPHABET.index(ch)
    # convert n to bytes (big-endian)
    full = n.to_bytes((n.bit_length() + 7)//8, "big") if n else b"\x00"
    # deal with leading '1's => leading zero bytes
    pad = 0
    for ch in b58s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00"*pad + full

def wif_to_privkey(wif: str) -> bytes:
    raw = b58decode(wif)
    if len(raw) not in (37, 38):  # 0x80 + 32 + [0x01] + 4
        raise ValueError("Bad WIF length")
    payload, chk = raw[:-4], raw[-4:]
    if hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4] != chk:
        raise ValueError("Bad WIF checksum")
    if payload[0] != 0x80:
        raise ValueError("Bad WIF prefix")
    core = payload[1:]
    if len(core) == 33 and core[-1] == 0x01:
        core = core[:-1]
    if len(core) != 32:
        raise ValueError("Bad WIF payload")
    return core

def rpc_call(method, params):
    r = requests.post(RPC, json={"jsonrpc":"2.0","id":1,"method":method,"params":params})
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]

def pack_varuint(n: int) -> bytes:
    out = b""
    while True:
        b_ = n & 0x7F
        n >>= 7
        out += struct.pack("B", b_ | (0x80 if n else 0))
        if not n:
            return out

def pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return pack_varuint(len(b)) + b

def serialize_comment_op(p: dict) -> bytes:
    jm = p["json_metadata"]
    if not isinstance(jm, str):
        jm = json.dumps(jm, separators=(",", ":"))
    return (
        pack_varuint(COMMENT_OP_ID) +
        pack_string(p["parent_author"]) +
        pack_string(p["parent_permlink"]) +
        pack_string(p["author"]) +
        pack_string(p["permlink"]) +
        pack_string(p["title"]) +
        pack_string(p["body"]) +
        pack_string(jm)
    )

def serialize_tx(tx: dict) -> bytes:
    exp_secs = calendar.timegm(time.strptime(tx["expiration"], "%Y-%m-%dT%H:%M:%S"))
    out = b""
    out += struct.pack("<H", tx["ref_block_num"])
    out += struct.pack("<I", tx["ref_block_prefix"])
    out += struct.pack("<I", exp_secs)
    ops = tx["operations"]
    out += pack_varuint(len(ops))
    for name, payload in ops:
        assert name == "comment"
        out += serialize_comment_op(payload)
    out += pack_varuint(0)  # extensions
    return out

def sig_low_s(r: int, s: int):
    if s > N_CURVE // 2:
        s = N_CURVE - s
    return r, s

def sign_compact_recoverable(priv32: bytes, digest32: bytes) -> bytes:
    sk = SigningKey.from_string(priv32, curve=SECP256k1, hashfunc=hashlib.sha256)
    # RFC6979 deterministic signature on the given digest
    sig_str = sk.sign_digest_deterministic(digest32, sigencode=eutil.sigencode_string)
    r, s = eutil.sigdecode_string(sig_str, SECP256k1.order)
    # enforce low-s
    if s > SECP256k1.order // 2:
        s = SECP256k1.order - s
        sig_str = eutil.sigencode_string(r, s, SECP256k1.order)

    # choose recid by public key recovery
    cands = VerifyingKey.from_public_key_recovery_with_digest(
        sig_str, digest32, curve=SECP256k1,
        sigdecode=eutil.sigdecode_string, allow_truncate=False
    )
    my_vk = sk.verifying_key
    recid = next(i for i, vk in enumerate(cands) if vk.to_string() == my_vk.to_string())

    header = 27 + 4 + recid  # 31 or 32
    r_bytes = r.to_bytes(32, "big")
    s_bytes = s.to_bytes(32, "big")
    return bytes([header]) + r_bytes + s_bytes


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s or "post"

def build_comment_tx(author, posting_wif, parent_author, parent_permlink,
                     permlink, title, body, json_metadata=None, expire_seconds=60):
    import json
    if json_metadata is None:
        json_metadata = {}

    # --- ref block info
    dgp = rpc_call("condenser_api.get_dynamic_global_properties", [])
    head_num = dgp["head_block_number"]
    head_id  = bytes.fromhex(dgp["head_block_id"])
    ref_block_num    = head_num & 0xFFFF
    ref_block_prefix = struct.unpack_from("<I", head_id, 4)[0]

    # --- expiration (UTC)
    expiration = (datetime.now(timezone.utc) + timedelta(seconds=expire_seconds)).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")

    # --- IMPORTANT: stringify json_metadata for BOTH signing + broadcast
    jm_str = json.dumps(json_metadata, separators=(",", ":"))

    tx = {
        "ref_block_num": ref_block_num,
        "ref_block_prefix": ref_block_prefix,
        "expiration": expiration,
        "operations": [
            ["comment", {
                "parent_author": parent_author,
                "parent_permlink": parent_permlink,
                "author": author,
                "permlink": permlink,
                "title": title,
                "body": body,
                "json_metadata": jm_str   # <— string, not dict
            }]
        ],
        "extensions": []
    }

    # Serialize for signature (uses same jm_str internally)
    ser = serialize_tx(tx)
    digest = hashlib.sha256(CHAIN_ID + ser).digest()
    priv = wif_to_privkey(posting_wif)

    sig = sign_compact_recoverable(priv, digest)
    tx["signatures"] = [sig.hex()]
    return tx

def broadcast(tx: dict) -> dict:
    return requests.post(RPC, json={"jsonrpc":"2.0","id":1,"method":"condenser_api.broadcast_transaction_synchronous","params":[tx]}).json()

# ===== BUILD & BROADCAST TOP-LEVEL POST =====
permlink = f"{slugify(title)}-{int(time.time())}"
tx = build_comment_tx(
    author=ACCOUNT,
    posting_wif=POSTING_WIF,
    parent_author=parent_author,
    parent_permlink=parent_permlink,
    permlink=permlink,
    title=title,
    body=body,
    json_metadata=json_metadata
)
resp = broadcast(tx)
print(resp)
print("Your post permlink:", permlink)

# ===== Successful request response =====

#{'id': 1, 'jsonrpc': '2.0', 'result': {'block_num': 98398459, 'expired': False, 'id': '15b7b7de21b183f4ee42f6fd3409f7b3a823120e', 'rc_cost': 1476909538, 'trx_num': 25}}
#Your post permlink: testing-app-posting-1754783338

# ===== To retrieve the posted data see below =====

import requests

RPC = "https://api.hive.blog"

def rpc(method, params):
    r = requests.post(RPC, json={"jsonrpc":"2.0","id":1,"method":method,"params":params})
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["result"]

author   = "wanttoknow"
permlink = "testing-app-posting-1754783338"

c = rpc("condenser_api.get_content", [author, permlink])

print("Author:", c["author"])
print("Permlink:", c["permlink"])
print("Created (UTC):", c["created"])
print("Is root post?:", c["parent_author"] == "")

print("\n--- BODY ---\n")
print(c["body"])

# ===== Successful request response =====

#Author: wanttoknow
#Permlink: testing-app-posting-1754783338
#Created (UTC): 2025-08-09T23:48:57
#Is root post?: True

#--- BODY ---

#This is a top-level post signed with pure python. If you are reading this, the test was successful.