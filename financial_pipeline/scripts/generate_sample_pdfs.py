"""
Generate Sample Financial PDFs for Testing

This script creates 12 realistic synthetic financial PDFs covering:
- 4 companies
- 3 fiscal years each (2021, 2022, 2023)
- Different document types (annual report, earnings release)

Run: python scripts/generate_sample_pdfs.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("reportlab not installed. Installing...")
    os.system("pip install reportlab -q")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors


# ── Sample company data ───────────────────────────────────────────────────────
COMPANIES = {
    "GlobalBank Corp": {
        2021: {"revenue": 48200, "net_income": 12300, "ebitda": 18500, "eps": 4.21,
               "total_assets": 890000, "total_debt": 145000, "equity": 78000, "ocf": 15200},
        2022: {"revenue": 52100, "net_income": 13800, "ebitda": 20100, "eps": 4.73,
               "total_assets": 925000, "total_debt": 138000, "equity": 84000, "ocf": 17800},
        2023: {"revenue": 55800, "net_income": 14900, "ebitda": 22300, "eps": 5.12,
               "total_assets": 970000, "total_debt": 132000, "equity": 91000, "ocf": 19200},
    },
    "TechFinance Ltd": {
        2021: {"revenue": 8900, "net_income": 1820, "ebitda": 2600, "eps": 2.14,
               "total_assets": 24000, "total_debt": 3200, "equity": 12000, "ocf": 2100},
        2022: {"revenue": 11200, "net_income": 2350, "ebitda": 3400, "eps": 2.76,
               "total_assets": 28000, "total_debt": 2900, "equity": 14500, "ocf": 2800},
        2023: {"revenue": 13700, "net_income": 2980, "ebitda": 4200, "eps": 3.51,
               "total_assets": 33000, "total_debt": 2500, "equity": 17200, "ocf": 3500},
    },
    "Meridian Energy plc": {
        2021: {"revenue": 29400, "net_income": 3200, "ebitda": 8900, "eps": 1.82,
               "total_assets": 145000, "total_debt": 42000, "equity": 38000, "ocf": 7200},
        2022: {"revenue": 38900, "net_income": 5100, "ebitda": 13200, "eps": 2.90,
               "total_assets": 158000, "total_debt": 39000, "equity": 42000, "ocf": 10800},
        2023: {"revenue": 32100, "net_income": 3800, "ebitda": 10500, "eps": 2.16,
               "total_assets": 162000, "total_debt": 41000, "equity": 44000, "ocf": 8900},
    },
    "PharmaCo International": {
        2021: {"revenue": 15600, "net_income": 3200, "ebitda": 5400, "eps": 6.12,
               "total_assets": 52000, "total_debt": 8900, "equity": 28000, "ocf": 4100},
        2022: {"revenue": 17200, "net_income": 3650, "ebitda": 6100, "eps": 6.98,
               "total_assets": 56000, "total_debt": 8200, "equity": 30500, "ocf": 4700},
        2023: {"revenue": 19800, "net_income": 4280, "ebitda": 7200, "eps": 8.21,
               "total_assets": 61000, "total_debt": 7500, "equity": 33800, "ocf": 5600},
    },
}


def create_annual_report_pdf(company: str, year: int, data: dict, output_path: str):
    """Generate a realistic annual report PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Cover page ────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=24, spaceAfter=20, textColor=colors.HexColor('#1a3a5c')
    )
    story.append(Paragraph(company, title_style))
    story.append(Paragraph(f"Annual Report {year}", styles['Title']))
    story.append(Paragraph(f"For the Fiscal Year Ended December 31, {year}", styles['Normal']))
    story.append(Spacer(1, 1*cm))

    # ── Financial highlights ──────────────────────────────────────────────────
    story.append(Paragraph("Financial Highlights", styles['Heading1']))
    story.append(Paragraph(
        f"We are pleased to report strong financial performance for fiscal year {year}. "
        f"{company} delivered revenue of ${data['revenue']:,}M, reflecting our continued "
        f"strategic execution and disciplined capital allocation.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.5*cm))

    # ── Income Statement ──────────────────────────────────────────────────────
    story.append(Paragraph("Consolidated Income Statement", styles['Heading2']))
    story.append(Paragraph(f"For the year ended December 31, {year} (USD millions)", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    income_data = [
        ["", f"FY{year}", f"FY{year-1}"],
        ["Revenue / Net Sales", f"${data['revenue']:,}", f"${int(data['revenue']*0.92):,}"],
        ["Cost of Revenue", f"${int(data['revenue']*0.62):,}", f"${int(data['revenue']*0.93*0.63):,}"],
        ["Gross Profit", f"${int(data['revenue']*0.38):,}", f"${int(data['revenue']*0.92*0.37):,}"],
        ["Operating Expenses", f"${int(data['revenue']*0.18):,}", f"${int(data['revenue']*0.92*0.19):,}"],
        ["Operating Income", f"${int(data['revenue']*0.20):,}", f"${int(data['revenue']*0.92*0.18):,}"],
        ["Interest Expense", f"(${int(data['total_debt']*0.035):,})", f"(${int(data['total_debt']*0.038):,})"],
        ["Income Before Tax", f"${int(data['net_income']*1.25):,}", f"${int(data['net_income']*1.15):,}"],
        ["Income Tax Expense", f"(${int(data['net_income']*0.25):,})", f"(${int(data['net_income']*0.23):,})"],
        ["NET INCOME", f"${data['net_income']:,}", f"${int(data['net_income']*0.88):,}"],
        ["EBITDA", f"${data['ebitda']:,}", f"${int(data['ebitda']*0.91):,}"],
        ["EPS (Diluted)", f"${data['eps']:.2f}", f"${data['eps']*0.89:.2f}"],
    ]

    income_table = Table(income_data, colWidths=[9*cm, 4*cm, 4*cm])
    income_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, -3), (-1, -3), colors.HexColor('#e8f0f7')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (0, -3), (-1, -3), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(income_table)
    story.append(Spacer(1, 0.8*cm))

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    story.append(Paragraph("Consolidated Balance Sheet", styles['Heading2']))
    story.append(Paragraph(f"As of December 31, {year} (USD millions)", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    balance_data = [
        ["ASSETS", f"FY{year}", "LIABILITIES & EQUITY", f"FY{year}"],
        ["Cash & Equivalents", f"${int(data['total_assets']*0.08):,}",
         "Short-term Debt", f"${int(data['total_debt']*0.2):,}"],
        ["Short-term Investments", f"${int(data['total_assets']*0.07):,}",
         "Accounts Payable", f"${int(data['revenue']*0.04):,}"],
        ["Accounts Receivable", f"${int(data['revenue']*0.15):,}",
         "Long-term Debt", f"${int(data['total_debt']*0.8):,}"],
        ["Other Current Assets", f"${int(data['total_assets']*0.05):,}",
         "Other Liabilities", f"${int(data['total_assets']*0.03):,}"],
        ["Property, Plant & Equip", f"${int(data['total_assets']*0.22):,}",
         "Total Liabilities", f"${data['total_assets'] - data['equity']:,}"],
        ["Intangible Assets", f"${int(data['total_assets']*0.12):,}",
         "Shareholders' Equity", f"${data['equity']:,}"],
        ["Other Long-term Assets", f"${int(data['total_assets']*0.46):,}",
         "", ""],
        ["TOTAL ASSETS", f"${data['total_assets']:,}",
         "TOTAL LIAB + EQUITY", f"${data['total_assets']:,}"],
    ]

    balance_table = Table(balance_data, colWidths=[5.5*cm, 3*cm, 5.5*cm, 3*cm])
    balance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
    ]))
    story.append(balance_table)
    story.append(Spacer(1, 0.8*cm))

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Cash Flow Statement", styles['Heading2']))
    story.append(Spacer(1, 0.3*cm))

    cf_data = [
        ["Cash Flow Summary (USD millions)", f"FY{year}", f"FY{year-1}"],
        ["Net Income", f"${data['net_income']:,}", f"${int(data['net_income']*0.88):,}"],
        ["Depreciation & Amortisation", f"${int((data['ebitda']-data['net_income'])*0.6):,}",
         f"${int((data['ebitda']-data['net_income'])*0.55):,}"],
        ["Changes in Working Capital", f"${int(data['net_income']*0.05):,}",
         f"(${int(data['net_income']*0.03):,})"],
        ["OPERATING CASH FLOW", f"${data['ocf']:,}", f"${int(data['ocf']*0.88):,}"],
        ["Capital Expenditures", f"(${int(data['ocf']*0.28):,})", f"(${int(data['ocf']*0.88*0.3):,})"],
        ["FREE CASH FLOW", f"${int(data['ocf']*0.72):,}", f"${int(data['ocf']*0.88*0.70):,}"],
        ["Investing Activities", f"(${int(data['ocf']*0.35):,})", f"(${int(data['ocf']*0.88*0.38):,})"],
        ["Financing Activities", f"(${int(data['ocf']*0.30):,})", f"(${int(data['ocf']*0.88*0.25):,})"],
        ["Net Change in Cash", f"${int(data['ocf']*0.35):,}", f"${int(data['ocf']*0.88*0.37):,}"],
    ]

    cf_table = Table(cf_data, colWidths=[9*cm, 4*cm, 4*cm])
    cf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a148c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
        ('FONTNAME', (0, 6), (-1, 6), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#ede7f6')),
        ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#d1c4e9')),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(cf_table)
    story.append(Spacer(1, 0.8*cm))

    # ── Key Ratios ────────────────────────────────────────────────────────────
    story.append(Paragraph("Key Financial Ratios", styles['Heading2']))
    net_margin = data['net_income'] / data['revenue'] * 100
    roe = data['net_income'] / data['equity'] * 100
    roa = data['net_income'] / data['total_assets'] * 100
    d_e = data['total_debt'] / data['equity']

    ratios_data = [
        ["Ratio", f"FY{year}", "Interpretation"],
        ["Net Profit Margin", f"{net_margin:.1f}%", "Net income as % of revenue"],
        ["Return on Equity (ROE)", f"{roe:.1f}%", "Profit generated per $ of equity"],
        ["Return on Assets (ROA)", f"{roa:.1f}%", "Efficiency of asset utilisation"],
        ["Debt-to-Equity", f"{d_e:.2f}x", "Financial leverage"],
        ["EBITDA Margin", f"{data['ebitda']/data['revenue']*100:.1f}%", "Operational profitability"],
        ["EPS (Diluted)", f"${data['eps']:.2f}", "Earnings per share"],
    ]

    ratios_table = Table(ratios_data, colWidths=[5*cm, 3*cm, 9*cm])
    ratios_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#bf360c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fbe9e7')]),
    ]))
    story.append(ratios_table)
    story.append(Spacer(1, 0.8*cm))

    # ── MD&A text ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Management Discussion & Analysis", styles['Heading2']))
    story.append(Paragraph(
        f"Revenue grew {((data['revenue'] / int(data['revenue'] * 0.92)) - 1)*100:.1f}% "
        f"year-over-year to ${data['revenue']:,}M, driven by volume growth across all segments. "
        f"Net income of ${data['net_income']:,}M represents a net margin of {net_margin:.1f}%, "
        f"improving from the prior year. EBITDA of ${data['ebitda']:,}M reflects an EBITDA margin "
        f"of {data['ebitda']/data['revenue']*100:.1f}%. "
        f"The balance sheet remains strong with total assets of ${data['total_assets']:,}M "
        f"and a debt-to-equity ratio of {d_e:.2f}x. "
        f"Operating cash flow of ${data['ocf']:,}M demonstrates robust cash generation.",
        styles['Normal']
    ))

    doc.build(story)
    print(f"Created: {output_path}")


def main():
    output_dir = Path("./data/pdfs/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for company, years in COMPANIES.items():
        for year, data in years.items():
            filename = f"{company.replace(' ', '_')}_{year}_Annual_Report.pdf"
            output_path = str(output_dir / filename)
            create_annual_report_pdf(company, year, data, output_path)
            count += 1

    print(f"\nGenerated {count} sample PDFs in {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*.pdf")):
        print(f"  - {f.name} ({f.stat().st_size // 1024}KB)")
    print("\nTo test the pipeline:")
    print("  python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
