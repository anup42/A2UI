import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.stage2_responses import _downloaded_asset_content_valid, _sanitize_response_media


class Stage2AssetValidationTests(unittest.TestCase):
    def test_rejects_html_saved_as_jpeg(self) -> None:
        ok, reason = _downloaded_asset_content_valid(
            "https://example.com/photo.jpg",
            "photo.jpg",
            "text/html; charset=utf-8",
            b"<!DOCTYPE html><html><body>blocked</body></html>",
        )

        self.assertFalse(ok)
        self.assertIn("HTML", reason)

    def test_accepts_jpeg_magic_for_jpeg_url(self) -> None:
        ok, reason = _downloaded_asset_content_valid(
            "https://example.com/photo.jpg",
            "photo.jpg",
            "image/jpeg",
            b"\xff\xd8\xff\xe0" + b"\x00" * 32,
        )

        self.assertTrue(ok, msg=reason)

    def test_accepts_svg_text_for_svg_url(self) -> None:
        ok, reason = _downloaded_asset_content_valid(
            "https://cdn.example.com/icon.svg",
            "icon.svg",
            "image/svg+xml",
            b"<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>",
        )

        self.assertTrue(ok, msg=reason)

    def test_sanitizes_detached_media_sections(self) -> None:
        text = "\n".join(
            [
                "Phone Comparison",
                "",
                "Feature table goes here.",
                "",
                "Images:",
                "- iPhone: https://loremflickr.com/1200/800/iphone13pro",
                "- Pixel: https://loremflickr.com/1200/800/pixel6pro",
                "",
                "Quick Actions",
                "- Apple specs: https://support.apple.com/kb/SP852",
            ]
        )

        cleaned = _sanitize_response_media(text)

        self.assertIn("Phone Comparison", cleaned)
        self.assertIn("Quick Actions", cleaned)
        self.assertNotIn("Images:", cleaned)
        self.assertNotIn("loremflickr.com", cleaned)

    def test_sanitizes_random_inline_media_image_but_keeps_icon(self) -> None:
        text = (
            "Option 1: Phone comparison\n"
            "Media: Image=https://loremflickr.com/1200/800/phone "
            "Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/phone.svg\n"
            "- Camera-focused option."
        )

        cleaned = _sanitize_response_media(text)

        self.assertNotIn("loremflickr.com", cleaned)
        self.assertIn("Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/phone.svg", cleaned)


if __name__ == "__main__":
    unittest.main()
