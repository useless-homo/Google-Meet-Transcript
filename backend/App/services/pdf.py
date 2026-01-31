from weasyprint import HTML

def generate_pdf(meeting, transcript, summary):
    html = f"""
    <h1>{meeting}</h1>
    <h2>Summary</h2>
    <p>{summary['summary']}</p>

    <h3>Action Items</h3>
    <ul>
      {''.join(f'<li>{a}</li>' for a in summary['action_items'])}
    </ul>

    <h3>Transcript</h3>
    <p>{transcript}</p>
    """

    HTML(string=html).write_pdf("output.pdf")
    return "output.pdf"
