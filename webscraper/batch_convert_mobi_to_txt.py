import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

MOBI_EXTS = {".mobi", ".azw", ".azw3"}

def which(cmd: str) -> str | None:
    return shutil.which(cmd)

def has_calibre() -> bool:
    return which("ebook-convert") is not None

def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def extract_text_from_html_files(html_paths: list[Path], min_par_len: int = 30) -> str:
    paragraphs = []
    for hp in html_paths:
        try:
            html = hp.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{2,}", "\n\n", text)
            parts = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_par_len]
            paragraphs.extend(parts)
        except Exception as e:
            print(f"  ⚠️  Skipping {hp.name}: {e}")
    return "\n\n".join(paragraphs)

def spine_order_from_opf(opf_path: Path) -> list[Path]:
    """
    Parse OPF to get spine reading order and return resolved HTML file paths.
    """
    try:
        xml = ET.parse(opf_path)
        root = xml.getroot()
        ns = {}
        # collect namespaces
        for k, v in root.attrib.items():
            if k.startswith("xmlns:"):
                ns[k.split(":", 1)[1]] = v
        # best-effort namespace handling
        def q(tag):
            # Try namespaced and non-namespaced variants
            for prefix in (ns.get("opf"), ns.get("package"), None):
                if prefix:
                    yield f"{{{prefix}}}{tag}"
            yield tag

        manifest = {}
        for tagname in q("manifest"):
            m = root.find(tagname)
            if m is not None:
                for item in list(m):
                    iid = item.attrib.get("id")
                    href = item.attrib.get("href")
                    if iid and href:
                        manifest[iid] = href
                break

        spine_ids = []
        for tagname in q("spine"):
            s = root.find(tagname)
            if s is not None:
                for itemref in list(s):
                    iref = itemref.attrib.get("idref")
                    if iref:
                        spine_ids.append(iref)
                break

        base = opf_path.parent
        htmls = []
        for iid in spine_ids:
            href = manifest.get(iid)
            if href:
                p = (base / href).resolve()
                htmls.append(p)
        # filter to existing files only
        htmls = [p for p in htmls if p.exists() and p.suffix.lower() in {".html", ".htm", ".xhtml"}]
        return htmls
    except Exception:
        return []

def discover_htmls_fallback(outdir: Path) -> list[Path]:
    # Fallback: alphabetical order if OPF spine isn’t available.
    return sorted([p for p in outdir.rglob("*") if p.suffix.lower() in {".html", ".htm", ".xhtml"}])

def convert_with_calibre(mobi_path: Path, out_txt: Path) -> bool:
    code, out, err = run(["ebook-convert", str(mobi_path), str(out_txt)])
    if code == 0 and out_txt.exists() and out_txt.stat().st_size > 0:
        return True
    print(f"  ⚠️  Calibre failed ({mobi_path.name}): {err.strip() or out.strip()}")
    return False

def convert_with_kindleunpack(mobi_path: Path, out_txt: Path, min_par_len: int = 30) -> bool:
    # Try kindleunpack CLI first, then `python -m kindleunpack`
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        outdir = tmpdir / "unpacked"
        outdir.mkdir(parents=True, exist_ok=True)

        kcli = which("kindleunpack")
        attempted = []
        if kcli:
            attempted.append([kcli, "-d", "-r", str(mobi_path), str(outdir)])
        attempted.append([sys.executable, "-m", "kindleunpack", "-d", "-r", str(mobi_path), str(outdir)])

        ok = False
        for cmd in attempted:
            code, out, err = run(cmd)
            if code == 0 and any(outdir.iterdir()):
                ok = True
                break

        if not ok:
            print(f"  ❌ KindleUnpack failed for {mobi_path.name}. "
                  f"Install it or use Calibre.\n     Tried: {attempted}")
            return False

        # Find OPF to get proper spine order
        opfs = list(outdir.rglob("*.opf"))
        html_paths = []
        if opfs:
            html_paths = spine_order_from_opf(opfs[0])
        if not html_paths:
            html_paths = discover_htmls_fallback(outdir)

        if not html_paths:
            print(f"  ❌ No HTML content found after unpacking {mobi_path.name}")
            return False

        txt = extract_text_from_html_files(html_paths, min_par_len=min_par_len)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(txt, encoding="utf-8")
        return True

def batch_mobi_to_txt(input_dir: str, output_dir: str, min_paragraph_len: int = 30, prefer_calibre: bool = True):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    mobi_files = [p for p in in_dir.rglob("*") if p.suffix.lower() in MOBI_EXTS]
    if not mobi_files:
        print(f"No MOBI/AZW files found under {in_dir}")
        return

    use_calibre = prefer_calibre and has_calibre()
    print(f"Found {len(mobi_files)} file(s). Using {'Calibre' if use_calibre else 'KindleUnpack'} path.")

    for mobi in mobi_files:
        rel = mobi.relative_to(in_dir)
        out_txt = out_dir / rel.with_suffix(".txt").name
        print(f"➡️  {mobi.name} → {out_txt.name}")

        try:
            if use_calibre:
                if convert_with_calibre(mobi, out_txt):
                    print(f"   ✅ Wrote {out_txt}")
                    continue
                else:
                    print("   ↪️ Falling back to KindleUnpack…")

            if convert_with_kindleunpack(mobi, out_txt, min_par_len=min_paragraph_len):
                print(f"   ✅ Wrote {out_txt}")
            else:
                print(f"   ❌ Failed: {mobi.name}")
        except Exception as e:
            print(f"   ❌ Error processing {mobi.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_mobi_to_txt.py <input_dir> <output_dir> [--no-calibre] [--min-par 30]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    use_calibre = True
    min_par = 30

    if "--no-calibre" in sys.argv:
        use_calibre = False
    if "--min-par" in sys.argv:
        try:
            idx = sys.argv.index("--min-par")
            min_par = int(sys.argv[idx + 1])
        except Exception:
            print("Invalid --min-par value; using default 30")

    batch_mobi_to_txt(input_dir, output_dir, min_paragraph_len=min_par, prefer_calibre=use_calibre)