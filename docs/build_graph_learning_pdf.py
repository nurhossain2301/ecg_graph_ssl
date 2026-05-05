from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ecg_beat_and_ibi_graph_learning.md"
OUTPUT = ROOT / "ecg_beat_and_ibi_graph_learning.pdf"


def flush_page(pdf, lines, page_num):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.955
    for text, style in lines:
        if style == "h1":
            size, weight, gap = 18, "bold", 0.034
        elif style == "h2":
            size, weight, gap = 13, "bold", 0.026
        elif style == "bullet":
            size, weight, gap = 10.5, "normal", 0.019
        else:
            size, weight, gap = 10.5, "normal", 0.019

        ax.text(
            0.08,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            va="top",
            ha="left",
            family="DejaVu Sans",
            wrap=False,
        )
        y -= gap

    ax.text(
        0.5,
        0.035,
        f"Page {page_num}",
        fontsize=9,
        va="center",
        ha="center",
        family="DejaVu Sans",
        color="#666666",
    )
    pdf.savefig(fig)
    plt.close(fig)


def markdown_to_lines(markdown):
    output = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            output.append(("", "body"))
            continue
        if line.startswith("# "):
            output.append((line[2:].strip(), "h1"))
            output.append(("", "body"))
            continue
        if line.startswith("## "):
            output.append(("", "body"))
            output.append((line[3:].strip(), "h2"))
            continue
        if line.startswith("### "):
            output.append(("", "body"))
            output.append((line[4:].strip(), "h2"))
            continue
        if line.startswith("- "):
            wrapped = textwrap.wrap(
                line[2:].strip(), width=88, subsequent_indent="  "
            )
            for i, part in enumerate(wrapped):
                prefix = "- " if i == 0 else "  "
                output.append((prefix + part, "bullet"))
            continue

        wrapped = textwrap.wrap(line, width=92)
        if not wrapped:
            output.append(("", "body"))
        for part in wrapped:
            output.append((part, "body"))
    return output


def build_pdf():
    markdown = SOURCE.read_text(encoding="utf-8")
    lines = markdown_to_lines(markdown)

    page = []
    page_num = 1
    current_height = 0.0
    max_height = 0.88

    def line_height(style):
        if style == "h1":
            return 0.041
        if style == "h2":
            return 0.032
        return 0.021

    with PdfPages(OUTPUT) as pdf:
        for text, style in lines:
            h = line_height(style)
            if page and current_height + h > max_height:
                flush_page(pdf, page, page_num)
                page_num += 1
                page = []
                current_height = 0.0

            page.append((text, style))
            current_height += h

        if page:
            flush_page(pdf, page, page_num)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
