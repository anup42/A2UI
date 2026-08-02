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
from pipeline.ir_formats import encode_express_completion


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
                        "root": {
                            "type": "Stack",
                            "props": {"direction": "vertical"},
                            "children": ["t1"],
                        },
                        "t1": {
                            "type": "Text",
                            "props": {"text": "Hello", "variant": "h2"},
                            "children": [],
                        },
                    },
                },
            }
            row["a2ui_express"] = encode_express_completion(row["genui_json"])
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

    def test_stage4_active_renderer_compiles_express_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            express = "<a2ui>\nroot=Text(\"Hello\")\n</a2ui>"
            genui_path = run_dir / "genui.jsonl"
            genui_path.write_text(
                json.dumps({"ui_id": "u_express", "a2ui_express": express}) + "\n" +
                json.dumps({"ui_id": "u_bad", "a2ui_express": '{"root":"root"}'}) + "\n",
                encoding="utf-8",
            )
            logger = logging.getLogger("stage4_active_test")
            run_stage4(
                genui_path=genui_path,
                output_dir=run_dir / "rendered",
                assets_dir=ROOT / "renderer" / "lit",
                server_root=ROOT,
                logger=logger,
                render_images=False,
                renderer_name="lit",
                payload_format="messages",
                render_log_filename="render.jsonl",
            )
            assert (run_dir / "rendered" / "u_express.html").exists()
            assert not (run_dir / "rendered" / "u_bad.html").exists()
            rows = [json.loads(line) for line in (run_dir / "render.jsonl").read_text(encoding="utf-8").splitlines()]
            bad = next(row for row in rows if row["ui_id"] == "u_bad")
            assert "express_payload_rejected" in bad["render"]["error"]

            render_rows = list(iter_jsonl(run_dir / "render.jsonl"))
            self.assertEqual(render_rows[0].get("renderer"), "lit")
            self.assertEqual(render_rows[0].get("payload_format"), "a2ui_v1_wire")
            self.assertEqual(render_rows[1].get("payload_format"), "a2ui_express_v1")


if __name__ == "__main__":
    unittest.main()
