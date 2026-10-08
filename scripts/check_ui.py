"""Browser smoke test against an isolated, already processed sample workspace."""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--chrome", action="store_true", help="Use installed Google Chrome")
    args = parser.parse_args()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "gallery_editor.cli", "serve", "--root", str(args.root), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            try:
                if httpx.get(url + "/api/health", trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Local server did not start")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.chrome else None, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            page.wait_for_selector(".card")
            assert page.locator(".card").count() == 3
            page.click("#style")
            expect(page.locator("#message")).to_contain_text("Deep album analysis complete", timeout=60000)
            assert page.locator("#clusters details").count() > 0
            page.screenshot(path=str(args.root / "reports" / "ui-album-desktop.png"), full_page=True)
            page.click("#process")
            expect(page.locator("#message")).to_contain_text("Processing finished", timeout=180000)
            page.locator(".card").first.click()
            page.wait_for_selector("#editor", state="visible")
            page.locator("#zoom").fill("2")
            page.locator("#zoom").dispatch_event("input")
            page.select_option("#view", "original")
            assert not page.locator("#editedFigure").is_visible()
            page.select_option("#view", "both")
            page.select_option("#aspect", "1:1")
            assert page.locator("#save").is_disabled()
            assert "REJECTED" in page.locator("#metrics").inner_text()
            page.click("#disableCrop")
            page.fill("#color-contrast", "11")
            page.click("#save")
            expect(page.locator("#message")).to_contain_text("Manual edit saved", timeout=60000)
            assert "contrast +11.00" in page.locator("#summary").inner_text()
            page.screenshot(path=str(args.root / "reports" / "ui-editor-desktop.png"), full_page=True)
            page.click("#reset")
            expect(page.locator("#message")).to_contain_text("Restored automatic version", timeout=60000)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(args.root / "reports" / "ui-mobile.png"), full_page=True)
            assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
            assert not errors, errors
            browser.close()
        print(
            "PASS: deep-analysis UI, processing, review, crop rejection, zoom/toggle, manual save/reset, mobile; no JS errors"
        )
    finally:
        process.terminate()
        process.communicate(timeout=10)


if __name__ == "__main__":
    main()
