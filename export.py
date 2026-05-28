from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import io
import datetime


def export_chat_to_pdf(messages: list, sources: list) -> bytes:
    """
    Convert chat history to a downloadable PDF report.
    Returns PDF as bytes.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    # ── Colors ────────────────────────────────────────────────────────────────
    DARK    = HexColor("#1a1a2e")
    TEAL    = HexColor("#1D9E75")
    GRAY    = HexColor("#5F5E5A")
    LIGHT   = HexColor("#F8F9FA")
    BORDER  = HexColor("#D3D1C7")

    # ── Styles ────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle("title",
        fontSize=20, fontName="Helvetica-Bold",
        textColor=DARK, spaceAfter=4, alignment=TA_CENTER
    )
    sub_style = ParagraphStyle("sub",
        fontSize=10, fontName="Helvetica",
        textColor=GRAY, spaceAfter=16, alignment=TA_CENTER
    )
    q_style = ParagraphStyle("question",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=white, spaceAfter=4,
        backColor=TEAL, leftIndent=8, rightIndent=8,
        borderPad=6
    )
    a_style = ParagraphStyle("answer",
        fontSize=10, fontName="Helvetica",
        textColor=DARK, spaceAfter=8, leading=15,
        leftIndent=8
    )
    source_style = ParagraphStyle("source",
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=GRAY, leftIndent=16, spaceAfter=4
    )
    section_style = ParagraphStyle("section",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=TEAL, spaceBefore=12, spaceAfter=6
    )

    def hr():
        return HRFlowable(
            width="100%", thickness=0.5,
            color=BORDER, spaceAfter=8, spaceBefore=8
        )

    # ── Build content ─────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("Research Assistant — Chat Export", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}",
        sub_style
    ))
    story.append(hr())

    # Sources section
    if sources:
        story.append(Paragraph("Sources Loaded", section_style))
        for src in sources:
            name = src.get('name', '?')
            typ  = src.get('type', '?')
            story.append(Paragraph(f"• {name} ({typ})", source_style))
        story.append(hr())

    # Conversation
    story.append(Paragraph("Conversation", section_style))
    story.append(Spacer(1, 8))

    qa_pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            question = messages[i]["content"]
            answer   = messages[i+1]["content"] if i+1 < len(messages) else ""
            msg_sources = messages[i+1].get("sources", []) if i+1 < len(messages) else []
            metrics     = messages[i+1].get("metrics", {}) if i+1 < len(messages) else {}
            qa_pairs.append((question, answer, msg_sources, metrics))
            i += 2
        else:
            i += 1

    for idx, (question, answer_text, msg_sources, metrics) in enumerate(qa_pairs, 1):
        # Question
        story.append(Paragraph(
            f"Q{idx}: {question}",
            q_style
        ))

        # Answer
        # Clean answer text for PDF (remove markdown bold)
        clean_answer = answer_text.replace("**", "").replace("*", "")
        story.append(Paragraph(clean_answer, a_style))

        # Metrics if available
        if metrics:
            metrics_text = (
                f"⏱ {metrics.get('total_time_s','?')}s total  |  "
                f"🔍 {metrics.get('retrieval_time_s','?')}s retrieval  |  "
                f"💰 ${metrics.get('estimated_cost_usd',0):.6f}"
            )
            story.append(Paragraph(metrics_text, source_style))

        # Sources used
        if msg_sources:
            for src in msg_sources:
                story.append(Paragraph(
                    f"📎 {src.get('source','?')} — {src.get('section','?')}",
                    source_style
                ))

        story.append(Spacer(1, 10))
        if idx < len(qa_pairs):
            story.append(hr())

    if not qa_pairs:
        story.append(Paragraph(
            "No conversation to export yet.",
            a_style
        ))

    # Footer
    story.append(hr())
    story.append(Paragraph(
        "Multi-Source AI Research Assistant · github.com/harinirajendran26/rag-research-assistant",
        ParagraphStyle("footer", fontSize=8, fontName="Helvetica",
                       textColor=GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()