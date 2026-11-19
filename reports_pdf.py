"""PDF summary report for a date range: totals, daily chart, and the violation evidence list."""

from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from event_store import event_folder, event_stats, query_events

INK = (30, 36, 48)
MUTED = (110, 118, 132)
HELMET = (31, 138, 101)
VIOLATION = (196, 64, 64)


def _text(value):
    """fpdf core fonts are latin-1 only."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def build_report(root, start, end, camera=None, max_items=60):
    stats = event_stats(root, start, end, camera)
    violations, total_violations = query_events(root, start, end, category="no_helmet", camera=camera,
                                                limit=max_items)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_title(f"Helmet report {start} to {end}")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.cell(0, 10, "Helmet compliance report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    scope = f"{start} to {end}" + (f"  -  camera: {camera}" if camera else "  -  all cameras")
    pdf.cell(0, 6, _text(scope), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated {datetime.now().astimezone().isoformat(timespec='minutes')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    totals = stats["totals"]
    rate = totals["false_positive_rate"]
    tiles = [("Events", totals["events"]), ("Helmet", totals["helmet"]), ("No helmet", totals["no_helmet"]),
             ("Confirmed", totals["confirmed"]), ("False positive", totals["false_positive"]),
             ("Needs review", totals["needs_review"]),
             ("False-positive rate", "-" if rate is None else f"{rate * 100:.0f}%")]
    width = (pdf.w - pdf.l_margin - pdf.r_margin) / 4
    for index, (label, value) in enumerate(tiles):
        if index and index % 4 == 0:
            pdf.ln(18)
        x, y = pdf.l_margin + (index % 4) * width, pdf.get_y()
        pdf.set_draw_color(220, 224, 230)
        pdf.rect(x + 1, y, width - 2, 16)
        pdf.set_xy(x + 3, y + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(width - 6, 4, label)
        pdf.set_xy(x + 3, y + 7)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*INK)
        pdf.cell(width - 6, 7, str(value))
    pdf.ln(22)

    days = stats["days"][-31:]
    if days:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Events per day", new_x="LMARGIN", new_y="NEXT")
        chart_w, chart_h = pdf.w - pdf.l_margin - pdf.r_margin, 45
        top = pdf.get_y()
        peak = max(day["events"] for day in days) or 1
        slot = chart_w / len(days)
        bar = max(1.0, slot * 0.7)
        for index, day in enumerate(days):
            x = pdf.l_margin + index * slot + (slot - bar) / 2
            helmet_h = chart_h * day["helmet"] / peak
            violation_h = chart_h * day["no_helmet"] / peak
            pdf.set_fill_color(*HELMET)
            if helmet_h:
                pdf.rect(x, top + chart_h - helmet_h, bar, helmet_h, "F")
            pdf.set_fill_color(*VIOLATION)
            if violation_h:
                pdf.rect(x, top + chart_h - helmet_h - violation_h, bar, violation_h, "F")
        pdf.set_draw_color(200, 204, 210)
        pdf.line(pdf.l_margin, top + chart_h, pdf.l_margin + chart_w, top + chart_h)
        pdf.set_xy(pdf.l_margin, top + chart_h + 1)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*MUTED)
        pdf.cell(chart_w / 2, 4, days[0]["date"])
        pdf.cell(chart_w / 2, 4, days[-1]["date"], align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*HELMET)
        pdf.cell(25, 5, "Helmet")
        pdf.set_text_color(*VIOLATION)
        pdf.cell(25, 5, f"No helmet  (peak {peak}/day)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "B", 12)
    shown = f" (first {len(violations)} of {total_violations})" if total_violations > len(violations) else ""
    pdf.cell(0, 8, f"No-helmet events{shown}", new_x="LMARGIN", new_y="NEXT")
    if not violations:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No no-helmet events in this period.", new_x="LMARGIN", new_y="NEXT")
    for event in violations:
        if pdf.get_y() > pdf.h - 45:
            pdf.add_page()
        y = pdf.get_y()
        try:
            image = event_folder(Path(root), event["date"], event["id"], event.get("category")) / "person.jpg"
        except ValueError:
            image = None
        if image and image.is_file():
            try:
                pdf.image(str(image), x=pdf.l_margin, y=y, h=26)
            except Exception:  # unreadable evidence must not break the report
                pass
        source = event.get("source") or {}
        detection = event.get("event_detection") or {}
        pdf.set_xy(pdf.l_margin + 32, y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, _text(f"{event['date']} {event['timestamp'][11:19]}  -  {event.get('status', '').replace('_', ' ')}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin + 32)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        camera_name = source.get("camera") or source.get("filename") or source.get("type") or "unknown"
        pdf.cell(0, 5, _text(f"Source: {camera_name}   Score: {detection.get('confidence', 0):.2f}   "
                             f"Plate: {event.get('plate_text') or 'not read'}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin + 32)
        pdf.cell(0, 5, _text(f"Event {event['id']}   Association: {event.get('association_status', '')}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*INK)
        pdf.set_y(y + 29)

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 4, "Detections are model suggestions. Only events marked 'confirmed' were checked by a person. "
                         "Plate-to-rider association is geometric and does not prove ownership.")
    return bytes(pdf.output())
