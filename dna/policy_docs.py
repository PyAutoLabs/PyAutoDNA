"""Decision pages: structured policy front matter and a small, safe markdown renderer.

A decision page may start with a YAML front-matter block (between ``---`` lines) naming the
policy it sets. The board reads those fields to show what is declared versus what is
recommended; the prose below the block is rendered as HTML. Everything is escaped: the renderer
emits only the tags it builds itself, and links only to http(s) URLs or sibling decision pages.
PyAutoDNA#6.
"""
import html
import re
from datetime import date, datetime
from pathlib import Path
import yaml

REPO_BLOB = 'https://github.com/PyAutoLabs/PyAutoDNA/blob/main/decisions/'
POLICY_KEYS = {'policy', 'title', 'status', 'declared_in', 'packages', 'recommended',
               'matrix_package', 'review_date', 'findings', 'links', 'supersedes'}


def split_front_matter(text):
    """``(fields or None, body)``. A page without front matter is prose only."""
    if not text.startswith('---\n'):
        return None, text
    end = text.find('\n---\n', 4)
    if end < 0:
        raise ValueError('front matter opened but never closed')
    fields = yaml.safe_load(text[4:end]) or {}
    if not isinstance(fields, dict):
        raise ValueError('front matter must be a mapping')
    return fields, text[end + 5:]


def validate_policy(fields, source='decision'):
    """Check a policy's fields; return them with ``review_date`` as an ISO string."""
    unknown = set(fields) - POLICY_KEYS
    if unknown:
        raise ValueError(f'{source}: unknown policy field(s) {sorted(unknown)}')
    for key in ('policy', 'title', 'status', 'declared_in', 'packages', 'recommended', 'review_date'):
        if not fields.get(key):
            raise ValueError(f'{source}: policy field {key!r} is required')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,60}', str(fields['policy'])):
        raise ValueError(f'{source}: policy id must be a short kebab-case name')
    packages = fields['packages']
    if not isinstance(packages, list) or not all(isinstance(p, str) for p in packages):
        raise ValueError(f'{source}: packages must be a list of names')
    recommended = fields['recommended']
    if not isinstance(recommended, dict) or set(recommended) != set(packages):
        raise ValueError(f'{source}: recommended must give a range for exactly the listed packages')
    from packaging.specifiers import SpecifierSet
    for spec in recommended.values():
        SpecifierSet(str(spec))
    review = fields['review_date']
    if isinstance(review, (date, datetime)):
        review = review.isoformat()[:10]
    datetime.strptime(str(review), '%Y-%m-%d')
    for key in ('findings', 'links'):
        if not isinstance(fields.get(key, []), list):
            raise ValueError(f'{source}: {key} must be a list')
    for link in fields.get('links', []):
        if not isinstance(link, dict) or not {'label', 'url'} <= set(link) or not safe_url(link['url']):
            raise ValueError(f'{source}: each link needs a label and an http(s) url')
    out = dict(fields, review_date=str(review))
    out.setdefault('matrix_package', packages[0])
    return out


def load_decisions(folder):
    """Every decision page: ``{'name', 'path', 'policy' (or None), 'body'}``, sorted by name."""
    docs = []
    for path in sorted(Path(folder).glob('*.md')):
        fields, body = split_front_matter(path.read_text())
        policy = validate_policy(fields, path.name) if fields else None
        docs.append({'name': path.stem, 'path': path, 'policy': policy, 'body': body})
    return docs


def safe_url(url):
    return isinstance(url, str) and re.fullmatch(r'https?://[^\s"<>]+', url) is not None


def _link_target(url):
    if safe_url(url):
        return url
    if re.fullmatch(r'[A-Za-z0-9_.-]+\.md(#[A-Za-z0-9_-]+)?', url):
        return REPO_BLOB + url  # a sibling decision page
    return None


def inline(text):
    """Escape, then add `code`, **bold** and [label](url) links."""
    parts = re.split(r'(`[^`]+`)', text)
    out = []
    for part in parts:
        if part.startswith('`') and part.endswith('`') and len(part) > 1:
            out.append('<code>' + html.escape(part[1:-1]) + '</code>')
            continue
        chunk = html.escape(part, quote=False)

        def link(m):
            target = _link_target(html.unescape(m.group(2)))
            label = m.group(1)
            if not target:
                return label
            return '<a href="' + html.escape(target) + '">' + label + '</a>'

        chunk = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', link, chunk)
        chunk = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', chunk)
        out.append(chunk)
    return ''.join(out)


def render_markdown(text, heading_offset=2):
    """Headings, paragraphs, bullet / numbered lists (with indented continuations), pipe
    tables, blockquotes and fenced code. Anything else is an escaped paragraph."""
    lines = text.splitlines()
    out, i = [], 0

    def heading(level, content):
        level = min(6, level + heading_offset)
        return f'<h{level}>' + inline(content) + f'</h{level}>'

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i])
                i += 1
            i += 1
            out.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
            continue
        m = re.match(r'(#{1,6})\s+(.*)', stripped)
        if m:
            out.append(heading(len(m.group(1)), m.group(2)))
            i += 1
            continue
        if stripped.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head, *rest = rows
                out.append('<div class="dna-table" role="region" tabindex="0"><table><thead><tr>'
                           + ''.join('<th scope="col">' + inline(c) + '</th>' for c in head)
                           + '</tr></thead><tbody>'
                           + ''.join('<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in r) + '</tr>' for r in rest)
                           + '</tbody></table></div>')
            continue
        if stripped.startswith('>'):
            quote = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append('<blockquote><p>' + inline(' '.join(quote)) + '</p></blockquote>')
            continue
        bullet = re.match(r'(\s*)([-*]|\d+\.)\s+(.*)', line)
        if bullet:
            ordered = bullet.group(2)[0].isdigit()
            items = []
            while i < len(lines):
                m = re.match(r'(\s*)([-*]|\d+\.)\s+(.*)', lines[i])
                if m and len(m.group(1)) <= len(bullet.group(1)):
                    items.append(m.group(3))
                elif lines[i].strip() and lines[i].startswith(' ') and items:
                    items[-1] += ' ' + lines[i].strip()  # continuation or nested line
                else:
                    break
                i += 1
            tag = 'ol' if ordered else 'ul'
            out.append(f'<{tag}>' + ''.join('<li>' + inline(it) + '</li>' for it in items) + f'</{tag}>')
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r'\s*(#{1,6}\s|[-*]\s|\d+\.\s|\||>|```)', lines[i]):
            para.append(lines[i].strip())
            i += 1
        if not para:  # a line the block rules declined (e.g. a lone marker): keep it as text
            para, i = [stripped], i + 1
        out.append('<p>' + inline(' '.join(para)) + '</p>')
    return '\n'.join(out)
