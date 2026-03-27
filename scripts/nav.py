"""Shared site navigation for all pages."""

NAV_CSS = """<style>
.site-nav,
.site-nav *,
.site-nav *::before,
.site-nav *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
.site-nav {
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 32px 0;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  font-weight: 400;
  line-height: 1.625;
  color: #2c2c2c;
}
@media (max-width: 768px) {
  .site-nav { padding: 24px 16px 0; }
}
.site-nav-title {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1.6rem;
  font-weight: 400;
  color: #2c2c2c;
  line-height: 1.2;
}
@media (max-width: 640px) {
  .site-nav-title { font-size: 1.3rem; }
}
.site-nav-subtitle {
  font-size: 0.8rem;
  color: #888;
  margin-top: 2px;
  line-height: 1.625;
}
.site-nav-links {
  margin-top: 12px;
  padding-bottom: 2px;
  font-size: 0.82rem;
  line-height: 1.625;
}
.site-nav-links a {
  color: #888;
  text-decoration: underline;
  text-decoration-color: rgba(136, 136, 136, 0.5);
  text-underline-offset: 4px;
  transition: color 0.15s, text-decoration-color 0.15s;
}
.site-nav-links a:hover {
  color: #2c2c2c;
  text-decoration-color: #2c2c2c;
}
.site-nav-links a.active {
  color: #2c2c2c;
  font-weight: 600;
  text-decoration: none;
}
.site-nav-sep {
  color: #d5d0c8;
  margin: 0 8px;
}
.site-nav-hr {
  border: none;
  border-top: 1px solid #d5d0c8;
  margin: 16px 0 24px;
}
</style>"""

PLAUSIBLE_SCRIPT = (
    '<script async src="https://plausible.io/js/pa-9xq45uThwNCmhvlOMB0yM.js"></script>\n'
    "<script>\n"
    "  window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};\n"
    "  plausible.init()\n"
    "</script>"
)

PLAUSIBLE_EVENTS_SCRIPT = (
    "<script>\n"
    "function track(name,props){if(window.plausible)plausible(name,{props:props||{}})}\n"
    "var _trackTimers={};\n"
    "function trackDebounced(name,propsFn,ms){clearTimeout(_trackTimers[name]);"
    "_trackTimers[name]=setTimeout(function(){track(name,propsFn())},ms||500)}\n"
    "</script>"
)

PAGES = [("Explorer", "index.html"), ("Analysis", "analysis.html"), ("Map", "map.html"), ("About", "about.html")]


def nav_html(active: str) -> str:
    """Return the site header + nav HTML. *active* is 'explorer', 'analysis', or 'map'."""
    links = []
    for label, href in PAGES:
        cls = ' class="active"' if label.lower() == active else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    sep = '<span class="site-nav-sep">&middot;</span>'
    return f"""<div class="site-nav">
  <h1 class="site-nav-title">Claude Code Changelog</h1>
  <p class="site-nav-subtitle">Track every change to Claude Code</p>
  <nav class="site-nav-links">{sep.join(links)}</nav>
  <hr class="site-nav-hr">
</div>"""
