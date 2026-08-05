#!/usr/bin/env python3
"""
digitize_journal.py - Transcribe photographed journal pages into Markdown notes.

The pages never leave your computer: transcription happens through a local
vision-language model served over an OpenAI-compatible HTTP API (e.g. llama.cpp).
This script only talks to the address in `vision_base` (default: your own machine).

Pipeline (one request per page, no accumulating context):
    JPEG -> ffmpeg (EXIF-rotate + downscale) -> base64 data-URI
         -> local vision server /v1/chat/completions -> transcription
         -> Markdown note with frontmatter in your vault

Resume-safe: pages whose `source` already appears in an output note are skipped.
The date shown in the note filename is extracted from the transcription text
(photo timestamps do not match the journal date); if no date can be trusted the
page falls back to page-NNN.md and is left for your review.

Usage:
    python3 digitize_journal.py            # process pending pages
    python3 digitize_journal.py --check    # pre-flight: ffmpeg + config + server
    python3 digitize_journal.py --force    # reprocess everything
    python3 digitize_journal.py --config path/to/config.json
"""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"
DEFAULT_YEAR_RANGE = (1900, 2100)

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

SYSTEM_PROMPT = (
    "Transcribe this photograph of a handwritten journal page. "
    "Transcribe ALL text on the page verbatim and completely:\n"
    "- Keep the author's exact words; do NOT paraphrase, summarize, or fix grammar or spelling.\n"
    "- Preserve line breaks, paragraphs, bullet points, numbered lists, indentation, headers, "
    "dates, times, and margin notes as closely as reasonable, using light Markdown.\n"
    "- Include any printed planner text (headings, day/date labels) where present.\n"
    "- Anything you cannot transcribe with confidence MUST be flagged: write [illegible] for a "
    "word or short phrase, [unreadable region] for a larger area. Never silently skip or guess text.\n"
    "- Output ONLY the transcription, with no commentary, headings, or analysis."
)

LEGIBILITY_NOTE = (
    "\n\n> ⚠️ **This page was largely illegible — transcription is partial. "
    "Needs human examination.**\n"
)


def load_config(config_path):
    cfg = json.loads(config_path.read_text())
    for key in ("images_root", "vault_root", "temp_dir", "log_dir"):
        cfg[key] = Path(cfg[key]).expanduser()
    cfg["temp_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def resolve_ffmpeg(cfg):
    name = cfg.get("ffmpeg") or "ffmpeg"
    found = shutil.which(name)
    if found:
        return found
    return None


def setup_logger(cfg):
    log_dir = cfg["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / cfg.get("log_file", "digitize.log")
    logger = logging.getLogger("digitize")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("=== digitize_journal run %s ===", datetime.now().isoformat(timespec="seconds"))
    return logger, log_path


def collect_images(root):
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            imgs.append(p)
    imgs.sort(key=lambda p: p.relative_to(root).parts)
    return imgs


def preprocess(ffmpeg, src, dst, max_px):
    vf = (f"scale='min({max_px},iw)':'min({max_px},ih)':force_original_aspect_ratio=decrease")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vf", vf, "-q:v", "2", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def transcribe(cfg, img_bytes):
    img_b64 = base64.b64encode(img_bytes).decode()
    url = cfg["vision_base"].rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": cfg["model"],
        "temperature": 0.1,
        "max_tokens": cfg["max_tokens"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Transcribe this page."},
            ]},
        ],
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as r:
        resp = json.loads(r.read().decode())
    content = resp["choices"][0]["message"]["content"].strip()
    usage = resp.get("usage", {})
    finish = resp["choices"][0].get("finish_reason", "")
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), finish


def folder_years(rel_parts, year_range):
    lo, hi = year_range
    years = set()
    for part in rel_parts[:-1]:
        for m in re.finditer(r"(20\d{2})", part):
            years.add(int(m.group(1)))
        m2 = re.search(r"(20\d{2})\s*-\s*(20\d{2})", part)
        if m2:
            y1, y2 = int(m2.group(1)), int(m2.group(2))
            years.update(range(y1, y2 + 1))
    return years or set(range(lo, hi + 1))


def valid_ymd(y, mo, d, year_range):
    lo, hi = year_range
    if mo < 1 or mo > 12 or d < 1 or d > 31:
        return False
    if not lo <= y <= hi:
        return False
    try:
        datetime(y, mo, d)
        return True
    except ValueError:
        return False


def folder_start_hint(rel_dir, year_range):
    """Parse the leading month-day-year from a folder name like
    'December 27 2024 - August 06 2025' into (y, mo, d) or None."""
    m = re.search(
        r"(Jan\w*|Feb\w*|Mar\w*|Apr\w*|May|Jun\w*|Jul\w*|Aug\w*|Sep\w*|Oct\w*|Nov\w*|Dec\w*)"
        r"\w*\s+(\d{1,2})\w*,?\s+(20\d{2})",
        str(rel_dir))
    if not m:
        return None
    mo = MONTHS[m.group(1)[:3].lower()]
    d, y = int(m.group(2)), int(m.group(3))
    if valid_ymd(y, mo, d, year_range):
        return (y, mo, d)
    return None


def resolve_md(mo, d, years, ctx, year_range):
    """Resolve a month/day to a year. Single-year folder wins; otherwise carry
    forward from the running date in ctx."""
    if len(years) == 1:
        y = next(iter(years))
        return y if valid_ymd(y, mo, d, year_range) else None
    last = ctx.get("last_date") if ctx else None
    if not last:
        return None
    last_dt = datetime(*last)
    for y in (last[0], last[0] + 1):
        if not valid_ymd(y, mo, d, year_range):
            continue
        cand = datetime(y, mo, d)
        if last_dt <= cand and (cand - last_dt).days <= 45:
            return y
    return None


def extract_date(text, rel_parts, year_range, ctx=None):
    """Return 'YYYY-MM-DD', 'YYYY', or None for a journal page transcription.
    ctx (optional dict) carries the running date across pages in a folder."""
    years = folder_years(rel_parts, year_range)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:15]
    full = []      # (in_folder, line_idx, pos, y, mo, d)  - complete date incl. year
    month_day = []  # month+day resolved via folder/carry-forward year
    year_only = []  # (in_folder, line_idx, pos, y)

    # Strict M-D token: standalone, not part of a longer digit/word/slash/dash run.
    md_re = r"(?<![0-9A-Za-z./-])(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])(?![0-9A-Za-z./-])"

    for li, ln in enumerate(lines):
        for m in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", ln):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if valid_ymd(y, mo, d, year_range):
                full.append((y in years, li, m.start(), y, mo, d))
        for m in re.finditer(
                r"(?<![0-9A-Za-z])(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](20\d{2})(?![0-9A-Za-z])", ln):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            mo, d = (b, a) if a > 12 and b <= 12 else (a, b)
            if valid_ymd(y, mo, d, year_range):
                full.append((y in years, li, m.start(), y, mo, d))
        for m in re.finditer(
                r"(?<![0-9A-Za-z])(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](\d{2})(?![0-9A-Za-z])", ln):
            a, b, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y = 2000 + yy if yy < 30 else 1900 + yy
            mo, d = (b, a) if a > 12 and b <= 12 else (a, b)
            if valid_ymd(y, mo, d, year_range):
                full.append((y in years, li, m.start(), y, mo, d))
        for m in re.finditer(
                r"\b(Jan\w*|Feb\w*|Mar\w*|Apr\w*|May|Jun\w*|Jul\w*|Aug\w*|Sep\w*|Oct\w*|Nov\w*|Dec\w*)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(20\d{2})?\b", ln):
            mo = MONTHS[m.group(1)[:3].lower()]
            d = int(m.group(2))
            if m.group(3):
                y = int(m.group(3))
                if valid_ymd(y, mo, d, year_range):
                    full.append((y in years, li, m.start(), y, mo, d))
            else:
                y = resolve_md(mo, d, years, ctx, year_range)
                if y:
                    month_day.append((y in years, li, m.start(), y, mo, d))
        for m in re.finditer(md_re, ln):
            a, b = int(m.group(1)), int(m.group(2))
            mo, d = (b, a) if a > 12 and b <= 12 else (a, b)
            y = resolve_md(mo, d, years, ctx, year_range)
            if y:
                month_day.append((y in years, li, m.start(), y, mo, d))
        for m in re.finditer(r"\b(20\d{2})\b", ln):
            y = int(m.group(1))
            lo, hi = year_range
            if lo <= y <= hi:
                year_only.append((y in years, li, m.start(), y))

    result = None
    for pool in (full, month_day):
        if pool:
            best = min(pool, key=lambda c: (not c[0], c[1], c[2]))
            _, _, _, y, mo, d = best
            result = f"{y:04d}-{mo:02d}-{d:02d}"
            break
    if result is None and year_only:
        best = min(year_only, key=lambda c: (not c[0], c[1], c[2]))
        result = f"{best[3]:04d}"
    if result and ctx is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", result):
        ctx["last_date"] = (int(result[:4]), int(result[5:7]), int(result[8:10]))
    return result


def assess_legibility(content):
    """Return True when the transcription is dominated by unreadable markers."""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        return True
    markers = sum(1 for ln in lines if ln in ("[illegible]", "[unreadable region]"))
    if markers >= 3 and markers / len(lines) > 0.25:
        return True
    if markers >= 1 and markers / len(lines) >= 0.5:
        return True
    if len(content) < 60:
        return True
    counts = {}
    for ln in lines:
        if len(ln) >= 5:
            counts[ln] = counts.get(ln, 0) + 1
    if counts and max(counts.values()) >= 3:
        return True
    return False


def write_output(cfg, rel_parts, page_no, date_key, content, src_rel, force_status=None):
    rel_dir = Path(*rel_parts[:-1])
    prefix = cfg.get("output_prefix", "")
    out_dir = cfg["vault_root"] / cfg["journal_subdir"] / prefix / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{date_key} page-{page_no:03d}.md" if date_key else f"page-{page_no:03d}.md"
    out_path = out_dir / name

    status = force_status or ("needs-review" if assess_legibility(content) else "transcribed")
    if status == "needs-review":
        content = LEGIBILITY_NOTE.strip("\n") + "\n" + content

    fm = {
        "date": date_key,
        "source": str(src_rel),
        "page": page_no,
        "folder": str(rel_dir),
        "status": status,
        "tags": "journal",
    }
    lines = ["---"]
    for k, v in fm.items():
        if v in (None, "", []):
            continue
        if isinstance(v, (list, dict)):
            lines.append(f"{k}:\n" + "".join(f"  - {i}\n" for i in v).rstrip())
        elif isinstance(v, str) and (":" in v or v.startswith(("201", "202", "203"))):
            lines.append(f'{k}: "{v}"')
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    md = "\n".join(lines) + "\n\n" + content + "\n"
    out_path.write_text(md)
    return out_path


def build_index(cfg, logger):
    journal = cfg["vault_root"] / cfg["journal_subdir"]
    if not journal.exists():
        return None
    groups = {}
    total = 0
    for p in sorted(journal.rglob("*.md")):
        if p.name.startswith("_Index"):
            continue
        rel = p.relative_to(journal)
        key = str(rel.parent)
        groups.setdefault(key, []).append(p)
        total += 1

    lines = ["# Journal Index", "",
             f"Automatically generated by digitize_journal.py - {datetime.now().strftime('%Y-%m-%d')}.",
             f"{total} transcribed page(s).", ""]
    for key in sorted(groups):
        lines.append(f"## {key}  ({len(groups[key])})")
        for p in groups[key]:
            stem = p.stem
            display = stem
            lines.append(f"- [[{key}/{stem}|{display}]]")
        lines.append("")
    idx = journal / "_Index.md"
    idx.write_text("\n".join(lines))
    return idx


def check_setup(cfg, logger):
    """Pre-flight check: ffmpeg, required config, images, and vision server."""
    ok = True

    ffmpeg = resolve_ffmpeg(cfg)
    if ffmpeg:
        logger.info("OK   ffmpeg: %s", ffmpeg)
    else:
        logger.error("FAIL ffmpeg not found. Install ffmpeg or set \"ffmpeg\" in your config.")
        ok = False

    for key in ("images_root", "vault_root", "journal_subdir", "vision_base", "model"):
        if not cfg.get(key):
            logger.error("FAIL config is missing required key: %s", key)
            ok = False

    if ok and cfg.get("images_root"):
        if not cfg["images_root"].exists():
            logger.error("FAIL images_root does not exist: %s", cfg["images_root"])
            ok = False
        else:
            imgs = collect_images(cfg["images_root"])
            logger.info("OK   images: %d photo(s) under %s", len(imgs), cfg["images_root"])
            if not imgs:
                logger.warning("WARN no jpg/jpeg/png images found under images_root")

    if ok and cfg.get("vision_base"):
        try:
            url = cfg["vision_base"].rstrip("/") + "/v1/models"
            with urllib.request.urlopen(url, timeout=10) as r:
                data = json.loads(r.read().decode())
            ids = [m.get("id") for m in data.get("data", [])]
            logger.info("OK   vision server reachable at %s (serves: %s)",
                        cfg["vision_base"], ids or "unknown")
        except Exception as e:
            logger.error("FAIL vision server at %s unreachable: %s", cfg["vision_base"], e)
            ok = False

    logger.info("check %s", "passed" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Digitize photographed journals locally.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="path to config.json (default: next to this script)")
    ap.add_argument("--check", action="store_true", help="pre-flight check only")
    ap.add_argument("--force", action="store_true", help="reprocess all pages")
    ap.add_argument("--limit", type=int, default=None, help="process at most N pages")
    ap.add_argument("--root", default=None, help="override images_root")
    ap.add_argument("--log", default=None, help="override log_file name")
    ap.add_argument("--prefix", default=None, help="override output_prefix subdir")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        print("Copy config.example.json to config.json and edit it.", file=sys.stderr)
        return 1

    cfg = load_config(config_path)
    if args.root:
        cfg["images_root"] = Path(args.root).expanduser()
    if args.log:
        cfg["log_file"] = args.log
    if args.prefix:
        cfg["output_prefix"] = args.prefix
    year_range = tuple(cfg.get("year_range", DEFAULT_YEAR_RANGE))
    logger, log_path = setup_logger(cfg)

    if args.check:
        return 0 if check_setup(cfg, logger) else 1

    ffmpeg = resolve_ffmpeg(cfg)
    if not ffmpeg:
        logger.error("ffmpeg not found; install it or set \"ffmpeg\" in your config.")
        return 1

    imgs = collect_images(cfg["images_root"])
    logger.info("Found %d images under %s", len(imgs), cfg["images_root"])

    # Resume: skip images whose source path is already present in output notes
    done_sources = set()
    journal = cfg["vault_root"] / cfg["journal_subdir"]
    if journal.exists():
        for p in journal.rglob("*.md"):
            text = p.read_text(errors="replace")
            m = re.search(r'^source:\s*"?(.+?)"?\s*$', text, re.M)
            if m:
                done_sources.add(m.group(1).strip().strip('"'))

    stats = {"skipped": 0, "ok": 0, "fail": 0}
    page_no = {}  # leaf-dir -> counter
    date_ctx = {}  # leaf-dir -> {"last_date": (y, mo, d)}

    for idx, src in enumerate(imgs, 1):
        rel_parts = tuple(src.relative_to(cfg["images_root"]).parts)
        src_rel = "/".join(rel_parts)
        leaf = str(Path(*rel_parts[:-1]))
        page_no[leaf] = page_no.get(leaf, 0) + 1
        ctx = date_ctx.setdefault(leaf, {})
        if "last_date" not in ctx:
            hint = folder_start_hint(leaf, year_range)
            if hint:
                ctx["last_date"] = hint

        if not args.force and src_rel in done_sources:
            logger.info("SKIP  %s (already digitized)", src_rel)
            stats["skipped"] += 1
            continue
        if args.limit and stats["ok"] + stats["fail"] >= args.limit:
            logger.info("Reached --limit=%d", args.limit)
            break

        tmp_path = cfg["temp_dir"] / f"page_{idx:03d}.jpg"
        try:
            preprocess(ffmpeg, src, tmp_path, cfg["max_image_px"])
            content, p_tok, c_tok, finish = transcribe(cfg, tmp_path.read_bytes())
            if not content:
                raise RuntimeError("empty transcription")
            date_key = extract_date(content, rel_parts, year_range, ctx)
            out = write_output(cfg, rel_parts, page_no[leaf], date_key, content, src_rel)
            extra = f" (TRUNCATED @ {c_tok} tok)" if finish == "length" else ""
            logger.info("OK    %s -> %s  [prompt=%d out=%d]%s",
                        src_rel, out.relative_to(cfg["vault_root"]), p_tok, c_tok, extra)
            stats["ok"] += 1
        except Exception as e:
            logger.error("FAIL  %s  (%s: %s)", src_rel, type(e).__name__, e)
            stats["fail"] += 1
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if cfg.get("write_index", True):
        idx = build_index(cfg, logger)
        if idx:
            logger.info("Index written: %s", idx)

    logger.info("Done: %d ok, %d failed, %d skipped. Log: %s",
                stats["ok"], stats["fail"], stats["skipped"], log_path)
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
