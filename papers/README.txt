Arogya AI - IEEE conference paper
=================================

UPLOAD TO OVERLEAF
  New Project -> Upload Project -> drop this .zip in.
  Set the compiler to pdfLaTeX (Menu -> Compiler).
  It compiles as-is. Nothing to install, nothing to upload separately.

CONTENTS
  arogya_ai.tex        the whole paper, single file
  figures/             every image the paper references
  make_figures.py      regenerates the 3 data charts
  make_ui_figures.py   regenerates the 5 interface figures

THE FIGURES

  Data charts - plotted from the paper's own tables, nothing invented:
    fig_differential.pdf   Table V   (ranked differential)
    fig_latency.pdf        Table III (before/after latency)
    fig_trace.pdf          Table VI  (per-stage timing)

  Interface figures - rendered from the real client source. Colours are the
  brand ramp in frontend/tailwind.config.js, the copy is the literal strings
  in frontend/src/components, and the Ayurvedic text is the Malaria row of
  rag/ayurveda_kb.csv:
    screenshot_chat.png        desktop chat + differential
    screenshot_emergency.png   red-flag emergency reply
    screenshot_result.png      triage + evidence + Ayurveda cards
    screenshot_trace.png       expanded per-turn trace
    screenshot_mobile.png      phone-width layout

  These are renderings, not photographic captures, and the paper says so in
  the caption of Fig. 3. If you would rather ship real captures, take them
  and overwrite the PNGs keeping the same filenames - the .tex needs no edit.

NOTE ON LENGTH
  With all 18 floats this runs about 9-10 pages. For a 6-page limit, drop
  screenshot_mobile, screenshot_trace and fig_trace first: their content is
  already carried by Table VI and the prose.
