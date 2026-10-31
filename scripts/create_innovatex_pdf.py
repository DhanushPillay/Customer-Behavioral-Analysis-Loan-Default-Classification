from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "InnovateX2_0_submission.pdf"
PAGE_W, PAGE_H = landscape((13.333 * inch, 7.5 * inch))

PURPLE = colors.HexColor("#71347D")
TEAL = colors.HexColor("#126F93")
NAVY = colors.HexColor("#172B4D")
INK = colors.HexColor("#292737")
PALE = colors.HexColor("#F4F0F6")
PALE_PURPLE = colors.HexColor("#D9C6DE")

styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=15,
    leading=21,
    textColor=INK,
    alignment=TA_LEFT,
    spaceAfter=7,
)
SMALL = ParagraphStyle(
    "small",
    parent=BODY,
    fontSize=11,
    leading=14,
)
TITLE = ParagraphStyle(
    "title",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=31,
    leading=34,
    textColor=PURPLE,
)


def draw_brand(c, slide_number, title, team="Banking DW Team"):
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Corner stripes echo the supplied InnovateX template.
    for offset, colour in ((0, NAVY), (0.28, TEAL), (0.56, PURPLE), (0.84, PALE_PURPLE)):
        c.setFillColor(colour)
        c.saveState()
        c.translate(0, PAGE_H - 0.05 * inch - offset * inch)
        c.rotate(-45)
        c.rect(-0.15 * inch, 0, 1.65 * inch, 0.23 * inch, fill=1, stroke=0)
        c.restoreState()

    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(1.58 * inch, PAGE_H - 0.58 * inch, team)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(1.58 * inch, PAGE_H - 1.14 * inch, title)
    c.setFillColor(TEAL)
    c.rect(1.48 * inch, PAGE_H - 1.42 * inch, 1.62 * inch, 0.055 * inch, fill=1, stroke=0)

    c.setFillColor(TEAL)
    c.setFont("Helvetica", 9)
    c.drawString(1.58 * inch, 0.34 * inch, "@InnovateX 2.0 Idea submission")
    c.setFillColor(PURPLE)
    c.drawRightString(PAGE_W - 1.85 * inch, 0.34 * inch, str(slide_number))
    c.setFillColor(PALE_PURPLE)
    c.rect(1.45 * inch, 0.08 * inch, PAGE_W - 3.0 * inch, 0.09 * inch, fill=1, stroke=0)
    c.setFillColor(PURPLE)
    c.rect(1.45 * inch, 0.08 * inch, (PAGE_W - 3.0 * inch) * 0.32, 0.09 * inch, fill=1, stroke=0)


def draw_logo(c):
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 23)
    c.drawRightString(PAGE_W - 0.45 * inch, PAGE_H - 0.66 * inch, "IEEE")
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 29)
    c.drawRightString(PAGE_W - 0.45 * inch, PAGE_H - 0.98 * inch, "EMBS")
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Oblique", 10)
    c.drawRightString(PAGE_W - 0.45 * inch, PAGE_H - 1.18 * inch, "Pune Chapter")


def draw_panel(c, x=1.48 * inch, y=1.02 * inch, width=10.4 * inch, height=4.05 * inch):
    c.setFillColor(PALE)
    c.setStrokeColor(PALE_PURPLE)
    c.setLineWidth(1)
    c.roundRect(x, y, width, height, 0.13 * inch, fill=1, stroke=1)


def draw_paragraph(c, text, x, y, width, style=BODY):
    para = Paragraph(text, style)
    _, height = para.wrap(width, PAGE_H)
    para.drawOn(c, x, y - height)
    return height


def bullet(c, text, x, y, width, style=BODY):
    c.setFillColor(INK)
    c.circle(x + 0.06 * inch, y - 0.13 * inch, 0.045 * inch, fill=1, stroke=0)
    return draw_paragraph(c, text, x + 0.25 * inch, y, width - 0.25 * inch, style)


def page(c, number, title):
    draw_brand(c, number, title)
    draw_logo(c)
    draw_panel(c)


def make_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("InnovateX 2.0 - Loan Default Risk Prediction")

    # 1. Title
    draw_brand(c, 1, "InnovateX 2.0")
    draw_logo(c)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1.58 * inch, PAGE_H - 1.92 * inch, "Loan Default Risk Prediction")
    c.setFillColor(TEAL)
    c.setFont("Helvetica", 14)
    c.drawString(1.58 * inch, PAGE_H - 2.25 * inch, "from Pre-Loan Transaction Behavior")
    draw_panel(c, y=1.28 * inch, height=3.35 * inch)
    fields = [
        "Problem Statement ID - Internal project",
        "Problem Statement Title - Loan Default Risk Prediction",
        "Theme - Data Mining and Predictive Analytics",
        "PS Category - Software",
        "Team ID - To be filled",
        "Team Name - To be filled",
    ]
    y = 4.22 * inch
    for item in fields:
        bullet(c, item, 1.92 * inch, y, 8.7 * inch, BODY)
        y -= 0.42 * inch
    c.showPage()

    # 2. Problem statement
    page(c, 2, "PROBLEM STATEMENT")
    draw_paragraph(c, "Banks lose money when loans default, even though useful warning signals may already exist in customer transaction history before the loan is granted.", 1.92 * inch, 4.65 * inch, 8.9 * inch, BODY)
    y = 3.88 * inch
    for text in [
        "The project predicts whether a loan will be repaid or default using only pre-loan transaction behavior.",
        "It uses 682 loans from the PKDD'99 Czech bank dataset: 606 repaid and 76 defaulted (11.1% default rate).",
        "The solution supports earlier, evidence-based risk screening and helps reduce avoidable credit losses.",
    ]:
        y -= bullet(c, text, 1.92 * inch, y, 9.25 * inch, BODY) + 0.09 * inch
    c.showPage()

    # 3. Objective
    page(c, 3, "OBJECTIVE")
    y = 4.58 * inch
    for text in [
        "Build a complete data-mining pipeline for loan default risk prediction.",
        "Engineer 13 leakage-checked behavioral features from each account's history before the loan date.",
        "Compare six classifiers with and without SMOTE, tune their hyperparameters, and select the most reliable model using repeated cross-validation.",
        "Translate model findings into practical risk indicators and customer behavior segments.",
    ]:
        y -= bullet(c, text, 1.92 * inch, y, 9.25 * inch, BODY) + 0.08 * inch
    c.showPage()

    # 4. Scope
    page(c, 4, "SCOPE")
    columns = [
        (1.92 * inch, "TARGET USERS", ["Credit analysts", "Risk managers", "Banking decision-makers"]),
        (5.00 * inch, "DATA AND APPLICATION", ["PKDD'99 financial dataset", "Pre-loan account activity", "Loan-level risk screening"]),
        (8.08 * inch, "KEY FEATURES", ["Preprocessing and EDA", "Classification and tuning", "Segmentation and dashboards"]),
    ]
    for x, heading, items in columns:
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, 4.45 * inch, heading)
        y = 3.98 * inch
        for item in items:
            y -= bullet(c, item, x, y, 2.55 * inch, SMALL) + 0.12 * inch
    c.setFillColor(INK)
    draw_paragraph(c, "The prototype is designed for analytical decision support; final lending decisions remain subject to banking policy and human review.", 1.92 * inch, 2.10 * inch, 8.9 * inch, SMALL)
    c.showPage()

    # 5. Proposed solution
    page(c, 5, "PROPOSED SOLUTION")
    c.setFillColor(INK)
    draw_paragraph(c, "A reproducible Python pipeline converts raw customer records into an interpretable loan-risk assessment.", 1.92 * inch, 4.66 * inch, 8.95 * inch, BODY)
    steps = [("01", "Raw data", "loan, transaction, account, client, disp"), ("02", "Features", "13 pre-loan behavior measures"), ("03", "Models", "6 classifiers + tuning + SMOTE"), ("04", "Decision support", "risk scores, patterns, segments")]
    x = 1.95 * inch
    for index, (num, heading, detail) in enumerate(steps):
        c.setFillColor(PURPLE if index % 2 == 0 else TEAL)
        c.roundRect(x, 2.35 * inch, 2.18 * inch, 1.15 * inch, 0.08 * inch, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x + 0.16 * inch, 3.16 * inch, num)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(x + 0.16 * inch, 2.84 * inch, heading)
        draw_paragraph(c, detail, x + 0.16 * inch, 2.68 * inch, 1.86 * inch, ParagraphStyle("step", parent=SMALL, textColor=colors.white, fontSize=9, leading=11))
        if index < len(steps) - 1:
            c.setFillColor(TEAL)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(x + 2.28 * inch, 2.74 * inch, ">")
        x += 2.68 * inch
    draw_paragraph(c, "Selected model: tuned Gradient Boosting (repeated-CV F1 0.571 +/- 0.120, Average Precision 0.654, ROC-AUC 0.864). Tuned Random Forest forms the leading group with ROC-AUC 0.877.", 1.92 * inch, 1.72 * inch, 9.0 * inch, SMALL)
    c.showPage()

    # 6. Technology stack
    page(c, 6, "TECHNOLOGY STACK")
    stack = [
        ("LANGUAGE", "Python 3.13.5"),
        ("DATA", "pandas, NumPy"),
        ("ML", "scikit-learn, imbalanced-learn"),
        ("VISUALIZATION", "Matplotlib, Seaborn"),
        ("WORKFLOW", "Jupyter, nbconvert"),
        ("OUTPUT", "CSV, PNG, PKL, dashboard")
    ]
    x_positions = [1.92 * inch, 5.15 * inch, 8.38 * inch]
    for index, (heading, detail) in enumerate(stack):
        x = x_positions[index % 3]
        y = 4.48 * inch if index < 3 else 2.85 * inch
        c.setFillColor(colors.white)
        c.setStrokeColor(PALE_PURPLE)
        c.roundRect(x, y - 0.68 * inch, 2.65 * inch, 0.95 * inch, 0.08 * inch, fill=1, stroke=1)
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 0.17 * inch, y - 0.07 * inch, heading)
        c.setFillColor(INK)
        c.setFont("Helvetica", 12)
        c.drawString(x + 0.17 * inch, y - 0.36 * inch, detail)
    c.setFillColor(INK)
    draw_paragraph(c, "Training-only preprocessing pipelines protect against leakage, while repeated stratified cross-validation supports honest model selection under class imbalance.", 1.92 * inch, 1.75 * inch, 8.95 * inch, SMALL)
    c.showPage()

    # 7. Innovation
    page(c, 7, "INNOVATION")
    y = 4.55 * inch
    for text in [
        "Uses transaction behavior that already exists before loan approval, rather than relying only on static demographics.",
        "Combines predictive classification with interpretable feature importance, risk quadrants, association patterns, and KMeans customer segmentation.",
        "Evaluates the honest minority-class metrics: F1, Average Precision, and ROC-AUC, instead of accuracy alone.",
        "Finds that low pre-loan balances and balance at loan time are stronger risk signals than age or gender, enabling focused and explainable screening.",
    ]:
        y -= bullet(c, text, 1.92 * inch, y, 9.25 * inch, BODY) + 0.08 * inch
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(1.92 * inch, 1.62 * inch, "Added value")
    c.setFillColor(INK)
    draw_paragraph(c, "A practical bridge between data mining coursework and a bank-ready analytical workflow: reproducible, measurable, and designed to support human decisions.", 1.92 * inch, 1.40 * inch, 8.9 * inch, SMALL)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    make_pdf()