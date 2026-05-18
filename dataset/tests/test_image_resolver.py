import unittest
from unittest.mock import patch

from pipeline import image_resolver


class DatasetImageResolverTests(unittest.TestCase):
    def test_travel_queries_prioritize_destination_specific_place(self):
        queries = image_resolver.travel_search_queries_from_row(
            {
                "area": "West Coast",
                "morning": "Arrive and settle in near Kata or Karon Beach",
                "afternoon": "Relax on the beach and enjoy a sunset walk",
            },
            "show four day vacation itinerary in phuket",
        )

        self.assertIn("Phuket", queries[0])
        self.assertIn("Kata", queries[0])
        self.assertIn("Beach", queries[0])
        self.assertNotIn("Arrive", queries[0])

    def test_flat_spec_travel_table_gets_commons_images(self):
        spec = {
            "root": "main",
            "state": {
                "days": [
                    {
                        "dayDate": "Day 1",
                        "area": "Kata Beach",
                        "morningActivity": "Beach walk",
                    }
                ]
            },
            "elements": {
                "main": {"type": "Stack", "props": {}, "children": ["table"]},
                "table": {
                    "type": "Table",
                    "props": {
                        "domain": "schedule",
                        "preferredPresentation": "cards",
                        "columns": [{"key": "dayDate", "label": "Day"}],
                        "statePath": "/days",
                    },
                    "children": [],
                },
            },
        }

        with patch.object(image_resolver, "search_commons_image_url", return_value="https://upload.wikimedia.org/example.jpg"):
            repaired, count = image_resolver.repair_flat_spec_images(
                spec,
                "show four day vacation itinerary in phuket",
                "Day 1: Kata Beach",
            )

        self.assertEqual(count, 1)
        row = repaired["state"]["days"][0]
        self.assertEqual(row["image"], "https://upload.wikimedia.org/example.jpg")
        self.assertIn("image", [col["key"] for col in repaired["elements"]["table"]["props"]["columns"]])
        self.assertIn("imageAlt", [col["key"] for col in repaired["elements"]["table"]["props"]["columns"]])

    def test_remote_image_component_gets_commons_fallback_and_replacement(self):
        spec = {
            "root": "main",
            "state": {},
            "elements": {
                "main": {"type": "Stack", "props": {}, "children": ["title", "heroImage"]},
                "title": {"type": "Text", "props": {"text": "Lalbagh Botanical Garden"}, "children": []},
                "heroImage": {
                    "type": "Image",
                    "props": {
                        "url": "https://example.com/missing-lalbagh.jpg",
                        "alt": "Lalbagh Botanical Garden",
                    },
                    "children": [],
                },
            },
        }

        with patch.object(image_resolver, "is_reachable_image_url", return_value=False), patch.object(
            image_resolver,
            "search_commons_image_url",
            return_value="https://upload.wikimedia.org/lalbagh.jpg",
        ):
            repaired, count = image_resolver.repair_flat_spec_images(
                spec,
                "show 4 day itinerary for vacation in bengaluru",
                "Lalbagh Botanical Garden",
            )

        self.assertEqual(count, 1)
        props = repaired["elements"]["heroImage"]["props"]
        self.assertEqual(props["url"], "https://upload.wikimedia.org/lalbagh.jpg")
        self.assertEqual(props["fallbackUrl"], "https://upload.wikimedia.org/lalbagh.jpg")


if __name__ == "__main__":
    unittest.main()
