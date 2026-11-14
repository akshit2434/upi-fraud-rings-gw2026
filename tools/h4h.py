"""Drive the signed-in Chrome window (started with --remote-debugging-port=9333) via Playwright CDP.
usage:
  h4h.py state                       # url, title, visible text excerpt, buttons/inputs
  h4h.py goto <url>
  h4h.py click "<text>"              # click a button/tab/link by visible text
  h4h.py shot <path.png>             # screenshot
  h4h.py upload <csv> --confirm      # set the file on the page's <input type=file> (UPLOADS FOR REAL)
"""
import sys
from playwright.sync_api import sync_playwright

CDP = 'http://127.0.0.1:9333'


def page_of(b):
    pages = [p for c in b.contexts for p in c.pages if p.url.startswith('http')]
    hs = [p for p in pages if 'hack4hack' in p.url]
    return (hs or pages)[-1]


def main():
    cmd = sys.argv[1]
    with sync_playwright() as pw:
        b = pw.chromium.connect_over_cdp(CDP)
        p = page_of(b)
        if cmd == 'goto':
            p.goto(sys.argv[2], wait_until='domcontentloaded'); p.wait_for_timeout(2500); cmd = 'state'
        if cmd == 'click':
            p.get_by_text(sys.argv[2], exact=False).first.click(); p.wait_for_timeout(2500); cmd = 'state'
        if cmd == 'shot':
            p.screenshot(path=sys.argv[2], full_page=False); print('saved', sys.argv[2]); return
        if cmd == 'upload':
            assert '--confirm' in sys.argv, 'refusing to upload without --confirm'
            inp = p.locator('input[type=file]')
            print('file inputs:', inp.count())
            inp.first.set_input_files(sys.argv[2]); p.wait_for_timeout(6000); cmd = 'state'
        if cmd == 'state':
            print('URL', p.url, '| TITLE', p.title())
            print('file inputs:', p.locator('input[type=file]').count())
            print('buttons:', [t.strip()[:40] for t in p.locator('button').all_inner_texts() if t.strip()][:40])
            txt = p.locator('body').inner_text()
            print('--- text ---'); n = [x for x in sys.argv[3:] if x.isdigit()]; print(txt[:int(n[0]) if n else 3000])


if __name__ == '__main__':
    main()
