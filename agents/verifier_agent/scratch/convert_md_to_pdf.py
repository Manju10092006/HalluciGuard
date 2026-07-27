import os
import re
import subprocess
import sys
import time
from pathlib import Path
import markdown

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1a1a1a;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #0f172a;
            font-weight: 700;
            line-height: 1.25;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }}
        h1 {{
            font-size: 24pt;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            color: #1e293b;
        }}
        h2 {{
            font-size: 18pt;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 6px;
            margin-top: 2em;
        }}
        h3 {{ font-size: 14pt; }}
        h4 {{ font-size: 12pt; }}
        code {{
            font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 9.5pt;
        }}
        pre {{
            background-color: #0f172a;
            color: #f8fafc;
            padding: 14px;
            border-radius: 6px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3b82f6;
            background-color: #eff6ff;
            margin: 1.5em 0;
            padding: 10px 16px;
            color: #1e3a8a;
            border-radius: 0 4px 4px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            page-break-inside: avoid;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #0f172a;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
            margin: 2em 0;
            background: #ffffff;
            padding: 15px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            page-break-inside: avoid;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 2em 0;
        }}
    </style>
</head>
<body>
    {content}
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""

def convert_md_to_html(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    
    def replace_mermaid(match):
        code = match.group(1).strip()
        return f'<pre class="mermaid">\n{code}\n</pre>'
    
    text = re.sub(r'```mermaid\s*\n(.*?)```', replace_mermaid, text, flags=re.DOTALL)
    
    html_body = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'toc', 'nl2br']
    )
    
    full_html = HTML_TEMPLATE.format(
        title=md_path.stem.replace('_', ' ').title(),
        content=html_body
    )
    
    html_path = md_path.with_suffix(".html")
    html_path.write_text(full_html, encoding="utf-8")
    return html_path

def convert_html_to_pdf(html_path: Path, pdf_path: Path):
    cmd = [
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path.absolute()}",
        html_path.absolute().as_uri()
    ]
    subprocess.run(cmd, check=True)

def main():
    workspace = Path(r"c:\Users\S.Manjunath Reddy\OneDrive\Music\Pictures\Videos\HalluciGuard")
    md_files = [
        workspace / "PROJECT_ENGINEERING_AUDIT_REPORT.md",
        workspace / "JUDGE_AGENT_TECHNICAL_DOCUMENTATION.md",
        workspace / "VERIFIER_AGENT_TECHNICAL_DOCUMENTATION.md",
        workspace / "VERIFIER_AGENT_EXECUTION_AND_VALIDATION_REPORT.md",
        workspace / "VERIFIER_RETRIEVAL_DEBUG_REPORT.md"
    ]
    
    for md_file in md_files:
        if md_file.exists():
            print(f"Converting {md_file.name} to HTML...")
            html_file = convert_md_to_html(md_file)
            pdf_file = md_file.with_suffix(".pdf")
            print(f"Printing {html_file.name} to PDF {pdf_file.name}...")
            convert_html_to_pdf(html_file, pdf_file)
            print(f"SUCCESS: {pdf_file.name} generated ({pdf_file.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
