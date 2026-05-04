# 08_u_000008_01 - Kyoto Travel Demo

## Query

**Query ID:** `q_000008`
**Intent:** `Travel`
**Tags:** `travel, itinerary, kyoto, cards`

Create a 2-day Kyoto itinerary for a couple visiting 2026-10-18 to 2026-10-19. Preferences: temples, food, calm walking routes, photography, budget JPY 40,000 excluding hotel. Must include day cards with times, places, estimated transport, costs, and action links. Candidate places: Fushimi Inari, Kiyomizu-dera, Nishiki Market, Gion, Arashiyama Bamboo Grove, Tenryu-ji, Kinkaku-ji, Philosopher Path. Keep it realistic with morning/afternoon/evening blocks.

## Response

# Kyoto Itinerary for a Couple: October 2026

This two-day plan for October 18-19, 2026, balances Kyoto's most iconic temples and serene walks with rich culinary experiences. The itinerary is designed for a calm pace, with ample time for photography, and groups sights geographically to minimize travel time and maximize enjoyment.

# Planning & Budget Context

This schedule focuses on Eastern, Western, and Southern Kyoto to create a relaxed and immersive experience. Your budget of **JPY 40,000** for two people is well-suited for the activities listed below.

*   **Estimated Total Cost (2 people):** JPY 32,000 - JPY 36,000
*   **Cost Breakdown:** Transport (~JPY 5,000), Food (~JPY 24,000), Admissions (~JPY 3,200).
*   **Transport Tip:** Consider purchasing a Kyoto City Bus & Subway 1-Day Pass (JPY 1,100 per person) for each day to simplify travel within the city. JR lines for Arashiyama and Fushimi Inari will be separate but are inexpensive.

---

# Day 1: Eastern Kyoto's Charm & Gion's Night (Oct 18)

### Morning: Kiyomizu-dera & Historic Higashiyama
**Time: 9:00 AM - 12:00 PM**
Media: Image=https://loremflickr.com/1200/800/kiyomizu-dera,temple Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/building.svg
Start your day at the magnificent Kiyomizu-dera Temple, famous for its wooden stage that juts out from the main hall. After visiting the temple, take a calm walk down the preserved historic streets of Sannenzaka and Ninenzaka, which are filled with traditional shops and teahouses—perfect for photography.
*   **Transport:** Bus from Kyoto Station to Gojo-zaka or Kiyomizu-michi stop.
*   **Cost:** Admission: JPY 400/person.
*   Action: [Button: View on Map] https://www.google.com/maps/place/Kiyomizu-dera

### Afternoon: Philosopher's Path & Nishiki Market
**Time: 1:00 PM - 5:00 PM**
Media: Image=https://loremflickr.com/1200/800/nishiki-market,food Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/shop.svg
Take a taxi or bus to the northern part of Higashiyama to walk a portion of the Philosopher's Path, a tranquil stone path along a canal. Afterwards, head to the heart of Kyoto's food scene, Nishiki Market. Known as "Kyoto's Kitchen," this is the perfect place for a late lunch, sampling everything from fresh seafood skewers to matcha-flavored sweets.
*   **Transport:** Bus to Ginkaku-ji for the path, then another bus to Shijo Karasuma for the market.
*   **Cost:** Food: ~JPY 3,000 - 4,000/person.
*   Action: [Button: Explore Nishiki Market] https://www.kyoto-nishiki.or.jp/en/

### Evening: Gion's Lantern-Lit Lanes
**Time: 6:00 PM onwards**
Media: Image=https://loremflickr.com/1200/800/gion,night Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/signpost-split.svg
As evening falls, explore the famous Gion district, Kyoto's geisha district. Stroll down the beautifully preserved Hanamikoji Street and the scenic Shirakawa Lane. The area is at its most atmospheric after dark when the lanterns are lit. Find a restaurant for dinner and enjoy the traditional ambiance.
*   **Transport:** Walking distance from Nishiki Market.
*   **Cost:** Dinner: ~JPY 5,000 - 7,000/person.
*   Action: [Button: Gion District Guide] https://kyoto.travel/en/thingstodo/entertainment/gion.html

---

# Day 2: Arashiyama's Serenity & Fushimi's Gates (Oct 19)

### Morning: Arashiyama Bamboo Grove & Tenryu-ji
**Time: 8:30 AM - 12:00 PM**
Media: Image=https://loremflickr.com/1200/800/arashiyama-bamboo-grove Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/tree.svg
Arrive early to experience the ethereal beauty of the Arashiyama Bamboo Grove before the crowds. The towering bamboo stalks create a unique, serene atmosphere ideal for a quiet walk and photos. Afterwards, visit the adjacent Tenryu-ji Temple, a UNESCO World Heritage site with a stunning landscape garden.
*   **Transport:** JR Sagano Line from Kyoto Station to Saga-Arashiyama Station.
*   **Cost:** Tenryu-ji Garden Admission: JPY 500/person.
*   Action: [Button: Learn about Tenryu-ji] https://www.tenryuji.com/en/

### Afternoon: Riverside Views & Lunch
**Time: 12:00 PM - 3:00 PM**
Media: Image=https://loremflickr.com/1200/800/togetsukyo-bridge,kyoto Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/water.svg
Walk from the bamboo grove to the iconic Togetsukyo Bridge ("Moon Crossing Bridge"), which spans the Katsura River. The area offers picturesque views of the forested mountainsides. Enjoy a relaxing lunch at one of the many restaurants overlooking the river.
*   **Transport:** Walking.
*   **Cost:** Lunch

Icons:
- Alarm: https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/alarm.svg
- Calendar Event: https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/calendar-event.svg

## Full IR

```json
{
  "root": "root_stack",
  "state": {},
  "elements": {
    "root_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "lg",
        "padding": 16
      },
      "children": [
        "title_text",
        "intro_text",
        "planning_card",
        "divider_1",
        "day1_title",
        "day1_morning_card",
        "day1_afternoon_card",
        "day1_evening_card",
        "divider_2",
        "day2_title",
        "day2_morning_card",
        "day2_afternoon_card"
      ]
    },
    "title_text": {
      "type": "Text",
      "props": {
        "text": "Kyoto Itinerary for a Couple: October 2026",
        "variant": "h2"
      },
      "children": []
    },
    "intro_text": {
      "type": "Text",
      "props": {
        "text": "This two-day plan for October 18-19, 2026, balances Kyoto's most iconic temples and serene walks with rich culinary experiences. The itinerary is designed for a calm pace, with ample time for photography, and groups sights geographically to minimize travel time and maximize enjoyment."
      },
      "children": []
    },
    "planning_card": {
      "type": "Card",
      "props": {},
      "children": [
        "planning_stack"
      ]
    },
    "planning_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md",
        "padding": 16
      },
      "children": [
        "planning_title",
        "planning_intro",
        "cost_table",
        "transport_tip"
      ]
    },
    "planning_title": {
      "type": "Text",
      "props": {
        "text": "Planning & Budget Context",
        "variant": "h2"
      },
      "children": []
    },
    "planning_intro": {
      "type": "Text",
      "props": {
        "text": "This schedule focuses on Eastern, Western, and Southern Kyoto to create a relaxed and immersive experience. Your budget of JPY 40,000 for two people is well-suited for the activities listed below."
      },
      "children": []
    },
    "cost_table": {
      "type": "Table",
      "props": {
        "columns": [
          {
            "key": "item",
            "label": "Item"
          },
          {
            "key": "cost",
            "label": "Cost"
          }
        ],
        "rows": [
          {
            "item": "Estimated Total (2 people)",
            "cost": "JPY 32,000 - JPY 36,000"
          },
          {
            "item": "Transport",
            "cost": "~JPY 5,000"
          },
          {
            "item": "Food",
            "cost": "~JPY 24,000"
          },
          {
            "item": "Admissions",
            "cost": "~JPY 3,200"
          }
        ],
        "domain": "generic"
      },
      "children": []
    },
    "transport_tip": {
      "type": "Text",
      "props": {
        "text": "Transport Tip: Consider purchasing a Kyoto City Bus & Subway 1-Day Pass (JPY 1,100 per person) for each day to simplify travel within the city. JR lines for Arashiyama and Fushimi Inari will be separate but are inexpensive."
      },
      "children": []
    },
    "divider_1": {
      "type": "Divider",
      "props": {},
      "children": []
    },
    "day1_title": {
      "type": "Text",
      "props": {
        "text": "Day 1: Eastern Kyoto's Charm & Gion's Night (Oct 18)",
        "variant": "h2"
      },
      "children": []
    },
    "day1_morning_card": {
      "type": "Card",
      "props": {},
      "children": [
        "day1_morning_stack"
      ]
    },
    "day1_morning_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md"
      },
      "children": [
        "day1_morning_image",
        "day1_morning_content_stack"
      ]
    },
    "day1_morning_image": {
      "type": "Image",
      "props": {
        "url": "../assets/r_000008_01_1_kiyomizu-dera_temple_android.png",
        "fit": "cover",
        "height": 180
      },
      "children": []
    },
    "day1_morning_content_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md",
        "padding": 16
      },
      "children": [
        "day1_morning_header_stack",
        "day1_morning_time",
        "day1_morning_desc",
        "day1_morning_details",
        "day1_morning_button"
      ]
    },
    "day1_morning_header_stack": {
      "type": "Stack",
      "props": {
        "direction": "horizontal",
        "gap": "md",
        "align": "center"
      },
      "children": [
        "day1_morning_icon",
        "day1_morning_title"
      ]
    },
    "day1_morning_icon": {
      "type": "Icon",
      "props": {
        "name": "../assets/r_000002_01_3_building.svg"
      },
      "children": []
    },
    "day1_morning_title": {
      "type": "Text",
      "props": {
        "text": "Morning: Kiyomizu-dera & Historic Higashiyama",
        "variant": "h3"
      },
      "children": []
    },
    "day1_morning_time": {
      "type": "Text",
      "props": {
        "text": "Time: 9:00 AM - 12:00 PM",
        "variant": "caption"
      },
      "children": []
    },
    "day1_morning_desc": {
      "type": "Text",
      "props": {
        "text": "Start your day at the magnificent Kiyomizu-dera Temple, famous for its wooden stage. After, walk down the preserved historic streets of Sannenzaka and Ninenzaka, filled with traditional shops and teahouses."
      },
      "children": []
    },
    "day1_morning_details": {
      "type": "Text",
      "props": {
        "text": "Transport: Bus from Kyoto Station to Gojo-zaka or Kiyomizu-michi stop.\nCost: Admission: JPY 400/person.",
        "variant": "body"
      },
      "children": []
    },
    "day1_morning_button": {
      "type": "Button",
      "props": {
        "label": "View on Map",
        "variant": "borderless"
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": "https://www.google.com/maps/place/Kiyomizu-dera"
          }
        }
      }
    },
    "day1_afternoon_card": {
      "type": "Card",
      "props": {},
      "children": [
        "day1_afternoon_stack"
      ]
    },
    "day1_afternoon_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md"
      },
      "children": [
        "day1_afternoon_image",
        "day1_afternoon_content_stack"
      ]
    },
    "day1_afternoon_image": {
      "type": "Image",
      "props": {
        "url": "../assets/r_000008_01_3_nishiki-market_food_android.png",
        "fit": "cover",
        "height": 180
      },
      "children": []
    },
    "day1_afternoon_content_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md",
        "padding": 16
      },
      "children": [
        "day1_afternoon_header_stack",
        "day1_afternoon_time",
        "day1_afternoon_desc",
        "day1_afternoon_details",
        "day1_afternoon_button"
      ]
    },
    "day1_afternoon_header_stack": {
      "type": "Stack",
      "props": {
        "direction": "horizontal",
        "gap": "md",
        "align": "center"
      },
      "children": [
        "day1_afternoon_icon",
        "day1_afternoon_title"
      ]
    },
    "day1_afternoon_icon": {
      "type": "Icon",
      "props": {
        "name": "../assets/r_000008_01_4_shop.svg"
      },
      "children": []
    },
    "day1_afternoon_title": {
      "type": "Text",
      "props": {
        "text": "Afternoon: Philosopher's Path & Nishiki Market",
        "variant": "h3"
      },
      "children": []
    },
    "day1_afternoon_time": {
      "type": "Text",
      "props": {
        "text": "Time: 1:00 PM - 5:00 PM",
        "variant": "caption"
      },
      "children": []
    },
    "day1_afternoon_desc": {
      "type": "Text",
      "props": {
        "text": "Walk a portion of the Philosopher's Path, then head to Nishiki Market. Known as 'Kyoto's Kitchen,' this is the perfect place for a late lunch, sampling everything from fresh seafood to matcha-flavored sweets."
      },
      "children": []
    },
    "day1_afternoon_details": {
      "type": "Text",
      "props": {
        "text": "Transport: Bus to Ginkaku-ji for the path, then another bus to Shijo Karasuma for the market.\nCost: Food: ~JPY 3,000 - 4,000/person.",
        "variant": "body"
      },
      "children": []
    },
    "day1_afternoon_button": {
      "type": "Button",
      "props": {
        "label": "Explore Nishiki Market",
        "variant": "borderless"
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": "https://www.kyoto-nishiki.or.jp/en/"
          }
        }
      }
    },
    "day1_evening_card": {
      "type": "Card",
      "props": {},
      "children": [
        "day1_evening_stack"
      ]
    },
    "day1_evening_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md"
      },
      "children": [
        "day1_evening_image",
        "day1_evening_content_stack"
      ]
    },
    "day1_evening_image": {
      "type": "Image",
      "props": {
        "url": "../assets/r_000008_01_5_gion_night_android.png",
        "fit": "cover",
        "height": 180
      },
      "children": []
    },
    "day1_evening_content_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md",
        "padding": 16
      },
      "children": [
        "day1_evening_header_stack",
        "day1_evening_time",
        "day1_evening_desc",
        "day1_evening_details",
        "day1_evening_button"
      ]
    },
    "day1_evening_header_stack": {
      "type": "Stack",
      "props": {
        "direction": "horizontal",
        "gap": "md",
        "align": "center"
      },
      "children": [
        "day1_evening_icon",
        "day1_evening_title"
      ]
    },
    "day1_evening_icon": {
      "type": "Icon",
      "props": {
        "name": "../assets/r_000006_01_4_signpost-split.svg"
      },
      "children": []
    },
    "day1_evening_title": {
      "type": "Text",
      "props": {
        "text": "Evening: Gion's Lantern-Lit Lanes",
        "variant": "h3"
      },
      "children": []
    },
    "day1_evening_time": {
      "type": "Text",
      "props": {
        "text": "Time: 6:00 PM onwards",
        "variant": "caption"
      },
      "children": []
    },
    "day1_evening_desc": {
      "type": "Text",
      "props": {
        "text": "Explore the famous Gion district, Kyoto's geisha district. Stroll down Hanamikoji Street and the scenic Shirakawa Lane. The area is at its most atmospheric after dark when the lanterns are lit."
      },
      "children": []
    },
    "day1_evening_details": {
      "type": "Text",
      "props": {
        "text": "Transport: Walking distance from Nishiki Market.\nCost: Dinner: ~JPY 5,000 - 7,000/person.",
        "variant": "body"
      },
      "children": []
    },
    "day1_evening_button": {
      "type": "Button",
      "props": {
        "label": "Gion District Guide",
        "variant": "borderless"
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": "https://kyoto.travel/en/thingstodo/entertainment/gion.html"
          }
        }
      }
    },
    "divider_2": {
      "type": "Divider",
      "props": {},
      "children": []
    },
    "day2_title": {
      "type": "Text",
      "props": {
        "text": "Day 2: Arashiyama's Serenity & Fushimi's Gates (Oct 19)",
        "variant": "h2"
      },
      "children": []
    },
    "day2_morning_card": {
      "type": "Card",
      "props": {},
      "children": [
        "day2_morning_stack"
      ]
    },
    "day2_morning_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md"
      },
      "children": [
        "day2_morning_image",
        "day2_morning_content_stack"
      ]
    },
    "day2_morning_image": {
      "type": "Image",
      "props": {
        "url": "../assets/r_000008_01_7_arashiyama-bamboo-grove_android.png",
        "fit": "cover",
        "height": 180
      },
      "children": []
    },
    "day2_morning_content_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md",
        "padding": 16
      },
      "children": [
        "day2_morning_header_stack",
        "day2_morning_time",
        "day2_morning_desc",
        "day2_morning_details",
        "day2_morning_button"
      ]
    },
    "day2_morning_header_stack": {
      "type": "Stack",
      "props": {
        "direction": "horizontal",
        "gap": "md",
        "align": "center"
      },
      "children": [
        "day2_morning_icon",
        "day2_morning_title"
      ]
    },
    "day2_morning_icon": {
      "type": "Icon",
      "props": {
        "name": "../assets/r_000002_01_2_tree.svg"
      },
      "children": []
    },
    "day2_morning_title": {
      "type": "Text",
      "props": {
        "text": "Morning: Arashiyama Bamboo Grove & Tenryu-ji",
        "variant": "h3"
      },
      "children": []
    },
    "day2_morning_time": {
      "type": "Text",
      "props": {
        "text": "Time: 8:30 AM - 12:00 PM",
        "variant": "caption"
      },
      "children": []
    },
    "day2_morning_desc": {
      "type": "Text",
      "props": {
        "text": "Arrive early to experience the ethereal beauty of the Arashiyama Bamboo Grove. Afterwards, visit the adjacent Tenryu-ji Temple, a UNESCO World Heritage site with a stunning landscape garden."
      },
      "children": []
    },
    "day2_morning_details": {
      "type": "Text",
      "props": {
        "text": "Transport: JR Sagano Line from Kyoto Station to Saga-Arashiyama Station.\nCost: Tenryu-ji Garden Admission: JPY 500/person.",
        "variant": "body"
      },
      "children": []
    },
    "day2_morning_button": {
      "type": "Button",
      "props": {
        "label": "Learn about Tenryu-ji",
        "variant": "borderless"
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": "https://www.tenryuji.com/en/"
          }
        }
      }
    },
    "day2_afternoon_card": {
      "type": "Card",
      "props": {},
      "children": [
        "day2_afternoon_stack"
      ]
    },
    "day2_afternoon_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md"
      },
      "children": [
        "day2_afternoon_image",
        "day2_afternoon_content_stack"
      ]
    },
    "day2_afternoon_image": {
      "type": "Image",
      "props": {
        "url": "../assets/r_000008_01_9_togetsukyo-bridge_kyoto_android.png",
        "fit": "cover",
        "height": 180
      },
      "children": []
    },
    "day2_afternoon_content_stack": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "md",
        "padding": 16
      },
      "children": [
        "day2_afternoon_header_stack",
        "day2_afternoon_time",
        "day2_afternoon_desc",
        "day2_afternoon_details"
      ]
    },
    "day2_afternoon_header_stack": {
      "type": "Stack",
      "props": {
        "direction": "horizontal",
        "gap": "md",
        "align": "center"
      },
      "children": [
        "day2_afternoon_icon",
        "day2_afternoon_title"
      ]
    },
    "day2_afternoon_icon": {
      "type": "Icon",
      "props": {
        "name": "../assets/r_000008_01_10_water.svg"
      },
      "children": []
    },
    "day2_afternoon_title": {
      "type": "Text",
      "props": {
        "text": "Afternoon: Riverside Views & Lunch",
        "variant": "h3"
      },
      "children": []
    },
    "day2_afternoon_time": {
      "type": "Text",
      "props": {
        "text": "Time: 12:00 PM - 3:00 PM",
        "variant": "caption"
      },
      "children": []
    },
    "day2_afternoon_desc": {
      "type": "Text",
      "props": {
        "text": "Walk from the bamboo grove to the iconic Togetsukyo Bridge ('Moon Crossing Bridge'), which spans the Katsura River. The area offers picturesque views of the forested mountainsides."
      },
      "children": []
    },
    "day2_afternoon_details": {
      "type": "Text",
      "props": {
        "text": "Transport: Walking.\nCost: Lunch",
        "variant": "body"
      },
      "children": []
    }
  }
}
```

## Slide-Ready Representative IR

```json
{
  "root": "kyoto_itinerary",
  "state": {
    "budgetRows": [
      {
        "item": "Estimated Total (2 people)",
        "cost": "JPY 32,000 - JPY 36,000"
      },
      {
        "item": "Transport",
        "cost": "~JPY 5,000"
      },
      {
        "item": "Food",
        "cost": "~JPY 24,000"
      },
      {
        "item": "Admissions",
        "cost": "~JPY 3,200"
      }
    ],
    "dayCards": [
      {
        "id": "d1_morning",
        "time": "Day 1 Morning",
        "title": "Kiyomizu-dera & Historic Higashiyama",
        "image": "../assets/r_000008_01_1_kiyomizu-dera_temple_android.png",
        "cost": "~JPY 1,000 admission + local transit",
        "cta": "View on Map",
        "url": "https://www.google.com/maps/place/Kiyomizu-dera"
      },
      {
        "id": "d2_morning",
        "time": "Day 2 Morning",
        "title": "Arashiyama Bamboo Grove & Tenryu-ji",
        "image": "../assets/r_000008_01_7_arashiyama-bamboo-grove_android.png",
        "cost": "~JPY 1,000 admission + transit",
        "cta": "Learn about Tenryu-ji",
        "url": "https://www.tenryuji.com/en/"
      }
    ]
  },
  "elements": {
    "kyoto_itinerary": {
      "type": "Stack",
      "props": {
        "direction": "vertical",
        "gap": "lg"
      },
      "children": [
        "title",
        "contextCard",
        "budgetTable",
        "dayCardTemplate",
        "globalActions"
      ]
    },
    "title": {
      "type": "Text",
      "props": {
        "text": "Kyoto Itinerary for a Couple: October 2026",
        "variant": "h2"
      },
      "children": []
    },
    "contextCard": {
      "type": "Card",
      "props": {},
      "children": [
        "contextText"
      ]
    },
    "contextText": {
      "type": "Text",
      "props": {
        "text": "Two calm days across Eastern Kyoto, Arashiyama, Gion, Nishiki Market, and Fushimi Inari within a JPY 40,000 activity budget.",
        "variant": "body"
      },
      "children": []
    },
    "budgetTable": {
      "type": "Table",
      "props": {
        "statePath": "/budgetRows",
        "columns": [
          {
            "key": "item",
            "label": "Item"
          },
          {
            "key": "cost",
            "label": "Cost"
          }
        ],
        "domain": "booking",
        "preferredPresentation": "table"
      },
      "children": []
    },
    "dayCardTemplate": {
      "type": "Card",
      "props": {},
      "repeat": {
        "statePath": "/dayCards",
        "key": "id"
      },
      "children": [
        "dayImage",
        "dayTime",
        "dayTitle",
        "dayCost",
        "dayButton"
      ]
    },
    "dayImage": {
      "type": "Image",
      "props": {
        "url": {
          "$item": "image"
        },
        "fit": "cover"
      },
      "children": []
    },
    "dayTime": {
      "type": "Text",
      "props": {
        "text": {
          "$item": "time"
        },
        "variant": "label"
      },
      "children": []
    },
    "dayTitle": {
      "type": "Text",
      "props": {
        "text": {
          "$item": "title"
        },
        "variant": "h3"
      },
      "children": []
    },
    "dayCost": {
      "type": "Text",
      "props": {
        "text": {
          "$item": "cost"
        },
        "variant": "body"
      },
      "children": []
    },
    "dayButton": {
      "type": "Button",
      "props": {
        "label": {
          "$item": "cta"
        }
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": {
              "$item": "url"
            }
          }
        }
      }
    },
    "globalActions": {
      "type": "Stack",
      "props": {
        "direction": "horizontal",
        "gap": "md"
      },
      "children": [
        "routeButton",
        "hotelButton"
      ]
    },
    "routeButton": {
      "type": "Button",
      "props": {
        "label": "Open Kyoto Route Map"
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": "https://www.google.com/maps/search/Kyoto+2+day+itinerary"
          }
        }
      }
    },
    "hotelButton": {
      "type": "Button",
      "props": {
        "label": "Find Hotels"
      },
      "children": [],
      "on": {
        "press": {
          "action": "openUrl",
          "params": {
            "url": "https://www.booking.com/city/jp/kyoto.html"
          }
        }
      }
    }
  }
}
```
