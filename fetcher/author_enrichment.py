from __future__ import annotations
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup, NavigableString, Tag
from models import Paper

log = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_BATCH_URL = f"{S2_API_BASE}/paper/batch"
S2_AUTHOR_URL = f"{S2_API_BASE}/author"


def enrich_authors(
    papers: list[Paper],
    api_key: str | None = None,
    timeout: int = 10,
    max_authors_per_paper: int = 5,
    llm_api_key: str = "",
    llm_base_url: str | None = None,
    llm_model: str = "",
) -> list[Paper]:
    arxiv_papers = [p for p in papers if not p.id.startswith("rss:")]
    if not arxiv_papers:
        log.info("No arXiv papers to enrich")
        return papers

    # Stage A: Fetch affiliations from arXiv HTML pages + LLM fallback
    _enrich_affiliations_from_arxiv_html(
        arxiv_papers, timeout,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )

    # Stage B: OpenAlex fallback for authors still missing affiliations
    try:
        from fetcher.openalex_enrichment import enrich_from_openalex
        enrich_from_openalex(arxiv_papers, timeout=timeout)
    except Exception as e:
        log.warning("OpenAlex enrichment failed: %s", e)

    # Stage C: Fetch h-index from Semantic Scholar
    try:
        paper_s2_data = _batch_fetch_papers(arxiv_papers, api_key, timeout)
    except Exception as e:
        log.warning("S2 batch fetch failed entirely: %s", e)
        return papers

    author_ids_to_fetch: set[str] = set()
    paper_author_map: dict[str, list[dict]] = {}

    for paper in arxiv_papers:
        s2_data = paper_s2_data.get(paper.id)
        if not s2_data or "authors" not in s2_data:
            continue
        s2_authors = s2_data["authors"]
        paper_author_map[paper.id] = s2_authors
        for s2_author in s2_authors[:max_authors_per_paper]:
            if s2_author.get("authorId"):
                author_ids_to_fetch.add(s2_author["authorId"])

    h_index_map = _batch_fetch_h_indices(author_ids_to_fetch, api_key, timeout)

    for paper in arxiv_papers:
        s2_data = paper_s2_data.get(paper.id)

        # Backfill paper-level metadata from S2
        if s2_data:
            if not paper.venue and s2_data.get("venue"):
                paper.venue = s2_data["venue"].strip()
            if paper.citation_count is None and s2_data.get("citationCount") is not None:
                paper.citation_count = s2_data["citationCount"]
            if not paper.tldr:
                tldr_obj = s2_data.get("tldr")
                if isinstance(tldr_obj, dict):
                    paper.tldr = tldr_obj.get("text")
            if not paper.doi:
                ext_ids = s2_data.get("externalIds") or {}
                paper.doi = ext_ids.get("DOI")

        # Author enrichment
        s2_authors = paper_author_map.get(paper.id, [])
        if not s2_authors:
            continue
        enriched = paper.authors_enriched or [
            {"name": name, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
            for name in paper.authors
        ]
        _merge_s2_into_enriched(enriched, s2_authors, h_index_map)
        paper.authors_enriched = enriched

    return papers


def _enrich_affiliations_from_arxiv_html(
    papers: list[Paper],
    timeout: int,
    llm_api_key: str = "",
    llm_base_url: str | None = None,
    llm_model: str = "",
):
    llm_pending: list[tuple[Paper, str]] = []

    for paper in papers:
        try:
            affiliations, raw_text = _fetch_arxiv_html_affiliations(paper.id, timeout)
            if not affiliations:
                if raw_text:
                    llm_pending.append((paper, raw_text))
                continue
            if not paper.authors_enriched:
                paper.authors_enriched = [
                    {"name": name, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
                    for name in paper.authors
                ]
            unique_affils = set(a for a in affiliations if a)
            is_team_paper = len(unique_affils) == 1 and len(affiliations) < len(paper.authors_enriched)
            if is_team_paper:
                inst = next(iter(unique_affils))
                for entry in paper.authors_enriched:
                    entry["affiliation"] = inst
            else:
                for idx, affil in enumerate(affiliations):
                    if idx < len(paper.authors_enriched) and affil:
                        paper.authors_enriched[idx]["affiliation"] = affil
            log.debug("Parsed affiliations for %s from arXiv HTML", paper.id)
        except Exception as e:
            log.debug("Failed to parse arXiv HTML for %s: %s", paper.id, e)
        time.sleep(0.5)

    llm_worthy = [
        (p, t) for p, t in llm_pending
        if _INST_KEYWORDS.search(t) or len(t) > 200
    ]
    if llm_worthy and llm_api_key:
        log.info(
            "LLM fallback: %d/%d papers have institution-like text in author section",
            len(llm_worthy), len(llm_pending),
        )
        _enrich_affiliations_via_llm(llm_worthy, llm_api_key, llm_base_url, llm_model)


def _enrich_affiliations_via_llm(
    pending: list[tuple[Paper, str]],
    api_key: str,
    base_url: str | None,
    model: str,
):
    """Batch-send raw author section text to LLM for affiliation extraction."""
    import anthropic

    log.info("LLM affiliation extraction for %d papers where HTML parsing failed", len(pending))

    BATCH = 5
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        entries = []
        for paper, raw_text in batch:
            author_names = ", ".join(paper.authors[:10])
            entries.append(
                f"Paper: {paper.id}\n"
                f"Authors: {author_names}\n"
                f"Author section text from HTML page:\n{raw_text[:1500]}"
            )

        prompt = (
            "Extract author affiliations from the arXiv HTML author section text below.\n"
            "For each paper, map author names to their institution/affiliation.\n"
            "ONLY extract information that is explicitly present in the provided text. "
            "Do NOT guess or use external knowledge.\n"
            "If no affiliation can be determined for an author, use null.\n"
            "If one institution applies to all authors, set it for everyone.\n\n"
            "Return a JSON array with one object per paper:\n"
            '{"paper_id": "...", "affiliations": {"AuthorName": "Institution or null"}}\n\n'
            "Return ONLY the JSON array.\n\n"
            + "\n---\n".join(entries)
        )

        try:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = anthropic.Anthropic(**kwargs)
            resp = client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in resp.content:
                if block.type == "text":
                    text = block.text
                    break
            log.info(
                "LLM affiliation tokens: input=%d, output=%d",
                resp.usage.input_tokens, resp.usage.output_tokens,
            )
            _apply_llm_affiliations(batch, text)
        except Exception as e:
            log.warning("LLM affiliation extraction failed for batch %d: %s", i, e)


def _apply_llm_affiliations(batch: list[tuple[Paper, str]], llm_response: str):
    """Parse LLM JSON response and apply affiliations to papers."""
    text = llm_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                results = json.loads(match.group())
            except json.JSONDecodeError:
                log.warning("Failed to parse LLM affiliation response")
                return
        else:
            log.warning("Failed to parse LLM affiliation response")
            return

    result_map = {}
    for item in results:
        pid = item.get("paper_id", "")
        affils = item.get("affiliations", {})
        if pid and isinstance(affils, dict):
            result_map[pid] = affils

    for paper, _ in batch:
        affils = result_map.get(paper.id, {})
        if not affils:
            continue
        if not paper.authors_enriched:
            paper.authors_enriched = [
                {"name": name, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
                for name in paper.authors
            ]
        all_same = len(set(v for v in affils.values() if v)) == 1 and any(affils.values())
        if all_same:
            inst = next(v for v in affils.values() if v)
            for entry in paper.authors_enriched:
                if not entry.get("affiliation"):
                    entry["affiliation"] = inst
        else:
            for entry in paper.authors_enriched:
                name = entry["name"]
                inst = affils.get(name)
                if inst and not entry.get("affiliation"):
                    entry["affiliation"] = inst
        log.debug("LLM extracted affiliations for %s: %s", paper.id, affils)


def _fetch_arxiv_html_affiliations(
    arxiv_id: str, timeout: int,
) -> tuple[list[str | None], str]:
    """Return (affiliations_list, raw_author_section_text).

    The raw text is non-empty when an ltx_authors div exists but parsing
    yielded no affiliations — callers can pass it to an LLM for extraction.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PaperTracker/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8")

    soup = BeautifulSoup(html, "html.parser")
    authors_div = soup.find("div", class_="ltx_authors")

    if authors_div:
        for parse_fn in (
            _parse_structured_affiliations,
            _parse_superscript_affiliations,
            _parse_is_with_affiliations,
            _parse_team_paper_affiliations,
        ):
            result = parse_fn(authors_div)
            if result and any(a is not None for a in result):
                return result, ""

        raw_text = authors_div.get_text(" ", strip=True)
        return [], raw_text

    flat = _parse_flat_text_affiliations(soup)
    return flat, ""


def _parse_structured_affiliations(authors_div: Tag) -> list[str | None]:
    """Pattern A: each author in separate ltx_creator with ltx_affiliation_institution."""
    creators = authors_div.find_all("span", class_="ltx_role_author")
    if len(creators) <= 1:
        return []
    result = []
    for creator in creators:
        inst = creator.find("span", class_="ltx_affiliation_institution")
        result.append(inst.get_text(strip=True) if inst else None)
    return result


def _parse_superscript_affiliations(authors_div: Tag) -> list[str | None]:
    """Pattern B: authors have <sup>1,2</sup>, affiliations listed after <br> as <sup>N</sup>text."""
    personnames = authors_div.find_all("span", class_="ltx_personname")
    if not personnames:
        return []

    authors: list[tuple[str, list[str]]] = []
    affil_map: dict[str, str] = {}

    for pn in personnames:
        # Collect text content, splitting at <br> to separate authors from affiliations
        # Everything before the first affiliation-style <sup>N</sup> after a <br> is author area
        br_found = False
        author_zone_nodes = []
        affil_zone_nodes = []

        for child in pn.children:
            if isinstance(child, Tag) and child.name == "br":
                br_found = True
                continue
            if br_found:
                affil_zone_nodes.append(child)
            else:
                author_zone_nodes.append(child)

        if not br_found:
            continue

        # Extract authors from author zone: look for bold spans or text with superscripts
        _extract_authors_from_zone(author_zone_nodes, authors)

        # Extract affiliation map from affil zone: <sup>N</sup>InstitutionText
        _extract_affil_map(affil_zone_nodes, affil_map)

    if not authors or not affil_map:
        return []

    result = []
    for name, nums in authors:
        affil = None
        for n in nums:
            if n in affil_map:
                affil = affil_map[n]
                break
        result.append(affil)
    return result


def _extract_authors_from_zone(nodes: list, authors: list[tuple[str, list[str]]]):
    """Extract (name, [footnote_numbers]) from author zone nodes."""
    for node in nodes:
        if isinstance(node, NavigableString):
            continue
        if not isinstance(node, Tag):
            continue

        # Authors can be in bold spans, tabular cells, or directly as text
        bold_spans = node.find_all("span", class_=re.compile(r"ltx_font_bold"))
        if bold_spans:
            for bs in bold_spans:
                _extract_single_author(bs, authors)
            continue

        # Tabular layout: authors in ltx_td cells
        td_cells = node.find_all("span", class_="ltx_td")
        if td_cells:
            for td in td_cells:
                bolds = td.find_all("span", class_=re.compile(r"ltx_font_bold"))
                if bolds:
                    for b in bolds:
                        _extract_single_author(b, authors)
                else:
                    # Plain text authors in td
                    _extract_plain_authors_from_element(td, authors)
            continue

        # Check if node itself is a bold author
        if "ltx_font_bold" in (node.get("class") or []):
            _extract_single_author(node, authors)
            continue

        # Tabular wrapper
        if "ltx_tabular" in " ".join(node.get("class") or []):
            bolds = node.find_all("span", class_=re.compile(r"ltx_font_bold"))
            if bolds:
                for b in bolds:
                    _extract_single_author(b, authors)
            else:
                _extract_plain_authors_from_element(node, authors)


def _extract_single_author(elem: Tag, authors: list[tuple[str, list[str]]]):
    """Extract one author name and footnote numbers from an element."""
    # Remove footnotemark spans (they confuse parsing)
    for fm in elem.find_all("span", class_="ltx_role_footnotemark"):
        fm.decompose()

    sups = elem.find_all("sup", class_="ltx_sup")
    nums = []
    for sup in sups:
        text = sup.get_text(strip=True)
        # Extract digits, skip symbols like *, †, ‡
        for part in re.split(r"[,\s]+", text):
            cleaned = re.sub(r"[^\d]", "", part)
            if cleaned:
                nums.append(cleaned)
        sup.decompose()

    name = elem.get_text(strip=True)
    name = re.sub(r"[,\s]+$", "", name).strip()
    name = re.sub(r"^[,\s]+", "", name).strip()
    if name and len(name) > 1:
        authors.append((name, nums))


def _extract_plain_authors_from_element(elem: Tag, authors: list[tuple[str, list[str]]]):
    """Extract authors from element containing inline text + <sup> tags."""
    # Walk children to find text+sup patterns: "Name<sup>1,2</sup>"
    current_name = ""
    for child in elem.descendants:
        if isinstance(child, NavigableString):
            parent = child.parent
            # Skip text inside <sup> tags
            if parent and parent.name == "sup":
                continue
            text = str(child)
            text = re.sub(r"[*†‡§∗¶]+", "", text)
            current_name += text
        elif isinstance(child, Tag) and child.name == "sup":
            # This sup follows a name — extract numbers and finalize the name
            sup_text = child.get_text(strip=True)
            nums = []
            for part in re.split(r"[,\s]+", sup_text):
                cleaned = re.sub(r"[^\d]", "", part)
                if cleaned:
                    nums.append(cleaned)
            name = current_name.strip().rstrip(",").strip()
            name = re.sub(r"[*†‡§∗¶\s]+$", "", name).strip()
            if name and len(name) > 1 and not name.startswith("@"):
                authors.append((name, nums))
            current_name = ""

    # Handle trailing name without a sup
    name = current_name.strip().rstrip(",").strip()
    name = re.sub(r"[*†‡§∗¶\s]+$", "", name).strip()
    if name and len(name) > 1 and not name.startswith("@"):
        authors.append((name, []))


def _extract_affil_map(nodes: list, affil_map: dict[str, str]):
    """Extract {number: institution} mapping from affiliation zone nodes."""
    # Concatenate all affiliation zone into one string, preserving <sup> markers
    raw_parts = []
    for node in nodes:
        if isinstance(node, NavigableString):
            raw_parts.append(str(node))
        elif isinstance(node, Tag):
            raw_parts.append(str(node))
    affil_html = "".join(raw_parts)

    # Flatten nested structures: extract all <sup> and text at any depth
    affil_soup = BeautifulSoup(affil_html, "html.parser")

    # Strategy: convert to text with <sup> markers, then split by numbered sups
    # Replace all <sup> with a unique delimiter
    for sup in affil_soup.find_all("sup"):
        num_text = sup.get_text(strip=True)
        num = re.sub(r"[^\d]", "", num_text)
        if num:
            sup.replace_with(f"\x00SUP{num}\x00")
        else:
            sup.decompose()

    # Remove footnote/email elements
    for elem in affil_soup.find_all("span", class_=re.compile(r"ltx_font_typewriter|ltx_role_footnote|ltx_note")):
        elem.decompose()

    flat_text = affil_soup.get_text()

    # Split by SUP markers: everything between SUPn and the next SUPm is institution n
    parts = re.split(r"\x00SUP(\d+)\x00", flat_text)
    # parts = [before_first_sup, num1, text1, num2, text2, ...]
    for i in range(1, len(parts) - 1, 2):
        num = parts[i]
        inst = parts[i + 1].strip()
        inst = re.sub(r"\s+", " ", inst)
        inst = re.sub(r"\s*[\w.+-]+@[\w.-]+\s*", "", inst)
        inst = inst.strip().rstrip(",;. ")
        if inst and len(inst) > 2:
            affil_map[num] = inst


def _parse_is_with_affiliations(authors_div: Tag) -> list[str | None]:
    """Pattern C: 'X is/are with Y' in author notes, or 'The authors are with Y'."""
    notes = authors_div.find("span", class_="ltx_author_notes")
    if not notes:
        return []

    notes_text = notes.get_text(" ", strip=True)
    names_raw = authors_div.find_all("span", class_="ltx_personname")
    if not names_raw:
        return []

    author_list = []
    for n in names_raw:
        text = n.get_text(strip=True)
        text = re.sub(r"[#†‡§∗*¶\d]+$", "", text).strip().rstrip(",").strip()
        for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", text):
            part = part.strip()
            if part:
                author_list.append(part)

    # Check for "The authors are with ..." (all same institution)
    global_m = re.search(
        r"(?:The\s+)?authors?\s+(?:is|are)\s+with\s+(?:the\s+)?(.+?)(?:\(|e-mail|\.\s|$)",
        notes_text, re.IGNORECASE,
    )
    if global_m:
        inst = global_m.group(1).strip().rstrip(",;. ")
        if inst and len(inst) > 2:
            return [inst] * len(author_list) if author_list else [inst]

    # Per-author mapping: "X is with Y"
    affiliation_map: dict[str, str] = {}
    for sentence in re.split(r"\.\s+|\n\s*\n", notes_text):
        m = re.match(r"(.+?)\s+(?:is|are)\s+with\s+(?:the\s+)?(.+)", sentence, re.IGNORECASE)
        if not m:
            continue
        name_part = m.group(1).strip()
        inst_part = m.group(2).strip()
        inst_name = inst_part.split(",")[0].strip() if inst_part else None
        if not inst_name:
            continue
        for mn in re.split(r",\s*(?:and\s+)?|\s+and\s+", name_part):
            mn = mn.strip()
            if mn:
                affiliation_map[mn] = inst_name

    if not affiliation_map:
        return []
    return [affiliation_map.get(name) for name in author_list]


_INST_KEYWORDS = re.compile(
    r"University|Institute|Lab(?:oratory)?|College|School|Corp|Inc|Ltd|Group|"
    r"Center|Centre|Research|Academy|Foundation|Department|Tech(?:nology)?|"
    r"Microsoft|Google|Meta|Alibaba|Tencent|Baidu|ByteDance|Huawei|NVIDIA|"
    r"Samsung|Intel|IBM|Amazon|Apple|DeepMind|OpenAI",
    re.IGNORECASE,
)


def _parse_team_paper_affiliations(authors_div: Tag) -> list[str | None]:
    """Pattern E: team papers where one ltx_role_author entry is actually an institution."""
    creators = authors_div.find_all("span", class_="ltx_role_author")
    if len(creators) < 2:
        return []

    names = []
    institutions = []
    for c in creators:
        text = c.get_text(strip=True)
        if not text:
            continue
        if _INST_KEYWORDS.search(text):
            institutions.append(text)
        else:
            names.append(text)

    if not institutions:
        return []

    inst = institutions[0]
    return [inst] * max(len(names), 1)


def _parse_flat_text_affiliations(soup: BeautifulSoup) -> list[str | None]:
    """Pattern D: no ltx_authors div, affiliations in flat paragraphs."""
    # Look for ltx_ERROR with {affiliations} — indicates broken LaTeX
    error_span = soup.find("span", class_="ltx_ERROR")
    if not error_span or "affiliation" not in error_span.get_text():
        return []

    # Find paragraphs near the error that contain institution-like text
    article = soup.find("article") or soup
    paras = article.find_all("div", class_="ltx_para", limit=10)

    institutions: list[str] = []
    for para in paras:
        text = para.get_text(" ", strip=True)
        if re.search(r"University|Institute|Lab|College|School|Corp|Inc|Ltd|Group|Center|Centre", text, re.IGNORECASE):
            # Extract institution lines separated by <br>
            for line in re.split(r"\n|<br", para.get_text("\n")):
                line = line.strip()
                if re.search(r"University|Institute|Lab|College|School|Corp|Inc|Ltd|Group|Center|Centre", line, re.IGNORECASE):
                    line = re.sub(r"\s*[\w.+-]+@[\w.-]+", "", line).strip()
                    if line:
                        institutions.append(line)
            break

    # Can't reliably map authors to institutions in this pattern
    # Return first institution for all authors as best guess
    if institutions:
        first_inst = institutions[0].strip().rstrip(",;. ")
        return [first_inst]
    return []


def _batch_fetch_papers(
    papers: list[Paper],
    api_key: str | None,
    timeout: int,
) -> dict[str, dict]:
    result = {}
    BATCH_SIZE = 100
    arxiv_ids = [f"ArXiv:{p.id}" for p in papers]

    for i in range(0, len(arxiv_ids), BATCH_SIZE):
        batch_ids = arxiv_ids[i:i + BATCH_SIZE]
        try:
            data = _s2_post(
                S2_BATCH_URL,
                params={"fields": "authors.authorId,authors.name,authors.affiliations,venue,citationCount,tldr,externalIds"},
                body={"ids": batch_ids},
                api_key=api_key,
                timeout=timeout,
            )
            if data and isinstance(data, list):
                for item, paper in zip(data, papers[i:i + BATCH_SIZE]):
                    if item is not None:
                        result[paper.id] = item
        except Exception as e:
            log.warning("S2 batch fetch failed for batch %d: %s", i, e)

        if i + BATCH_SIZE < len(arxiv_ids):
            time.sleep(1)

    return result


def _batch_fetch_h_indices(
    author_ids: set[str],
    api_key: str | None,
    timeout: int,
) -> dict[str, int]:
    result = {}
    for author_id in author_ids:
        try:
            url = f"{S2_AUTHOR_URL}/{author_id}"
            data = _s2_get(url, params={"fields": "hIndex"}, api_key=api_key, timeout=timeout)
            if data and "hIndex" in data and data["hIndex"] is not None:
                result[author_id] = data["hIndex"]
        except Exception as e:
            log.debug("S2 author fetch failed for %s: %s", author_id, e)
        time.sleep(0.1)
    return result


def _merge_s2_into_enriched(
    enriched: list[dict],
    s2_authors: list[dict],
    h_index_map: dict[str, int],
):
    for idx, s2_auth in enumerate(s2_authors):
        if idx >= len(enriched):
            enriched.append({
                "name": s2_auth.get("name", ""),
                "affiliation": None, "h_index": None, "semantic_scholar_id": None,
            })
        target = enriched[idx]
        author_id = s2_auth.get("authorId")
        if author_id:
            target["semantic_scholar_id"] = author_id
        s2_affiliations = s2_auth.get("affiliations") or []
        if not target.get("affiliation") and s2_affiliations:
            target["affiliation"] = s2_affiliations[0]
        if author_id and author_id in h_index_map:
            target["h_index"] = h_index_map[author_id]


def _s2_get(url: str, params: dict, api_key: str | None = None, timeout: int = 10) -> dict | None:
    from .retry import request_with_retry
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}" if query else url
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        data = request_with_retry(
            full_url,
            headers=headers,
            timeout=timeout,
            max_attempts=4,
            base_delay=5.0,
            max_delay=60.0,
        )
        return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except (TimeoutError, OSError):
        return None


def _s2_post(url: str, params: dict, body: dict, api_key: str | None, timeout: int) -> list | None:
    from .retry import request_with_retry
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}" if query else url
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        data = request_with_retry(
            full_url,
            method="POST",
            headers=headers,
            data=payload,
            timeout=timeout,
            max_attempts=4,
            base_delay=5.0,
            max_delay=60.0,
        )
        return json.loads(data.decode("utf-8"))
    except (TimeoutError, OSError):
        return None
