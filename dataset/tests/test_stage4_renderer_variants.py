import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.stage4_render import run_stage4
from pipeline.storage import iter_jsonl


class Stage4RendererVariantsTests(unittest.TestCase):
    def test_stage4_generates_primary_and_lit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            genui_path = run_dir / "genui.jsonl"
            row = {
                "ui_id": "u_test_01",
                "response_id": "r_test_01",
                "query_id": "q_test_01",
                "genui_json": {
                    "root": "root",
                    "state": {},
                    "elements": {
                        "root": {"type": "Column", "props": {}, "children": ["t1"]},
                        "t1": {
                            "type": "Text",
                            "props": {"text": "Hello", "variant": "h2"},
                            "children": [],
                        },
                    },
                },
            }
            genui_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            logger = logging.getLogger("stage4_test")
            logger.setLevel(logging.INFO)

            run_stage4(
                genui_path=genui_path,
                output_dir=run_dir / "rendered",
                assets_dir=ROOT / "renderer" / "json_render",
                server_root=ROOT,
                logger=logger,
                render_images=False,
                renderer_name="json_render",
                payload_format="flat_spec",
                render_log_filename="render.jsonl",
            )

            run_stage4(
                genui_path=genui_path,
                output_dir=run_dir / "rendered_lit",
                assets_dir=ROOT / "renderer" / "lit",
                server_root=ROOT,
                logger=logger,
                render_images=False,
                renderer_name="lit",
                payload_format="messages",
                render_log_filename="render_lit.jsonl",
            )

            self.assertTrue((run_dir / "rendered" / "u_test_01.html").exists())
            self.assertTrue((run_dir / "rendered_lit" / "u_test_01.html").exists())

            render_rows = list(iter_jsonl(run_dir / "render.jsonl"))
            lit_rows = list(iter_jsonl(run_dir / "render_lit.jsonl"))
            self.assertEqual(render_rows[0].get("renderer"), "json_render")
            self.assertEqual(lit_rows[0].get("renderer"), "lit")


if __name__ == "__main__":
    unittest.main()
