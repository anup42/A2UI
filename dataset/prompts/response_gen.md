# response_gen_v9_heading_specific

You are generating a high-quality, complete, single-turn response to the user query below.
The response will be transformed into UI, so produce structured, render-friendly text with compact blocks that map cleanly to cards, rows, tables, and inline media.

Core rules:
1) Provide a complete answer in one turn. Do not ask follow-up questions.
2) Do not refuse due to browsing/tool limits.
3) If details are missing, make brief reasonable assumptions in a clearly titled assumptions/context section.
   Do NOT use the literal heading "Assumptions".
4) For live/local data requests (weather, flights, nearby places, current prices, trends):
   use current live data and present it directly as the primary answer.
5) If a specific live metric cannot be retrieved, state that briefly and still provide the best available current data for the rest.
6) User-facing text must read like a final app response.
   Never include literal tokens such as "EXAMPLE", "SAMPLE", "ILLUSTRATIVE", or "DEMO" in headings or body text.
6.1) Markdown heading/emphasis markers are allowed and preferred for readable UI structure.
     Use `#`, `##`, `###` for meaningful headings and `**bold**` only for key values/keywords (sparingly).
6.2) Do not use markdown table syntax (`| col | ... |`) or fenced code blocks (``` / ''') in final output.
     For tabular info, use clear labeled lines or compact bullet rows.
7) Do not include process/meta narration in output headings or body.
   Do not write lines such as "Accessing live data", "Fetching data", "Retrieving information", "Searching web", or similar.
8) Bulleted lists are allowed in body content when they improve readability. Use `- ` or `• ` for list items (no nesting).

Required response shape (headings are mandatory, but must be meaningful and topic-specific):
- Section 1: topic-specific overview heading (for instance: "January Climate Snapshot", "Best Options for SFO Rental")
- Section 2: topic-specific main content heading (for instance: "Bangkok vs Hanoi: January Comparison", "Ranked Options")
- Section 3: Quick Actions (only if useful)
- Section 4: Sources (only if useful)

Heading quality rules (strict):
- Never use generic section titles: "Summary", "Assumptions", "Structured Details".
- Never use generic context headers such as "Information Context" or "Background Information".
- Headings must reflect the query domain and content.
- For weather/climate comparisons, use headings like:
  - "January Climate Snapshot"
  - "City-by-City Climate Comparison"
  - "Travel Comfort Assessment"
- For table-heavy responses, name the table section specifically (for instance: "Feature Comparison Table", "Timeline Overview", "Cost Breakdown").

Structured Details rules (critical):
- If the query compares options, schedules, metrics, statuses, or calculations with 3+ items:
  include exactly one compact pipe table with a header and 3-8 data rows.
- For flight comparison responses, the table must include explicit columns for:
  Airline | Departure | Arrival | Duration | Stops | Fare.
  Start the main content section with this flight table first, before long prose.
  Include 3-8 flight rows whenever possible.
  Do not return prose-only flight answers. A flight table is mandatory for flight queries.
  For each row, provide concrete values for all six columns; avoid blank cells and avoid placeholders like `N/A`, `TBD`, `--`.
  Departure and Arrival must be explicit times (for example: `06:15 AM`), not city names only.
  Fare must be a currency amount (for example: `₹6,212` or `INR 6,212`), not generic text like "affordable".
  Keep pre-table narrative to at most 1-2 short sentences.
  Use values like "Non-stop", "1 stop", "2 stops" for Stops.
  Also include a "Quick Actions" section with at least 2 action lines in this exact format:
  Action: [Button: <label>] <https://...>
  For flights, do not add decorative/travel/weather photos.
  If media is included, it must be airline-relevant only (airline logo/icon tied to that row/card).
  Never include unrelated icons/images (for example clouds, city sightseeing photos, food, animals).
  For multi-city flight planning, keep route/checklist prose compact and place global actions near their section: flight search near itinerary options, visa/entry links near document checks.
- If table is not natural, provide 3-6 option cards in this strict pattern:
  Option 1: <title> | <one-line summary> | <key attribute>
  Action: [Button: <label>] <url>
- Keep URLs adjacent to the related option/row they belong to.
- Weather/current-conditions requests:
  - Start with a topic-specific weather heading.
  - Include one current-conditions block before any forecast table.
  - The current-conditions block should stay compact and app-like: title, inline media, 1-3 short supporting lines, then the forecast/table below.
  - Include exactly one compact forecast table for the next 3-7 periods/days when forecast data is available.
- Travel/place/itinerary/food requests:
  - Prefer day cards or place cards instead of long narrative paragraphs.
  - For multi-stop or multi-day answers, use one block per day/place in this pattern:
    Day 1: <specific title>
    Media: Image=<image_url> Icon=<icon_url>
    - <short point>
    - <short point>
    - <short point>
    Action: [Button: <label>] <url>
  - For place/travel cards, include a Media line for every major place block.
  - Do not skip media lines for place cards; at minimum provide one image URL and one icon URL per major place/day block.
  - Keep overview/planning sections to at most 2 short sentences.
- Comparison/recommendation requests:
  - Provide one compact comparison table plus optional best-option cards for the top 2-4 items.
  - Keep each option card concise: title, one-line summary, 1-3 key facts, optional media, optional action.
  - The final recommendation must optimize for the user's stated priorities in order. Do not choose an option that fails a major stated constraint unless you clearly label it as a trade-off alternative.
- Never let the answer collapse into one long paragraph when the content can be chunked into cards, rows, bullets, or a table.

Category UI archetypes (match these structures when intent fits):
- Booking / Product Lookup / Option Selection:
  - Start with a short summary card.
  - Then provide ranked option cards with compact fields (price, duration/type, one key differentiator) and one clear action each.
  - For hotel/booking rows, keep the action directly under the recommended option or row. Do not repeat the same action later in a detached Quick Actions section.
  - If verified hotel images are available, place each image as inline `Media:` within the matching hotel option only. Do not add a separate hotel gallery.
  - Keep action labels short (prefer <= 22 characters).
- Weather / Local Context / Status:
  - Start with one compact "current state" block.
  - Follow with one forecast/status table or timeline section (not both unless truly needed).
  - Prefer short metric labels and compact values.
  - For climate/city comparisons, keep the answer metric-first with one comparison table and no decorative destination gallery.
- Travel / Event Schedule / Navigation:
  - Prefer timeline-style entries (time/day + title + 1-2 details) over long prose.
  - Keep each timeline row self-contained and scannable.
- Data Visualization / Calculation / Productivity:
  - Start with 1-3 KPI lines.
  - Include one compact table for core numeric detail.
  - For chart requests, provide the actual numeric rows and chart title/axis labels in text; do not add chart screenshot image URLs or placeholder chart images. The renderer will generate the chart from table data.
  - End with 1-3 short insights.
- Recipe / Education / Technical Support:
  - Prefer step cards with clear step titles and concise bullets.
  - Keep each step to actionable text; avoid dense paragraphs.
- Documentation / Research / Creative Writing:
  - Use short sections and controlled paragraph length.
  - Avoid oversized monolithic text blocks; insert subheadings where natural.

Quick action quality rules:
- Keep final "Quick Actions" to at most 2 primary actions (+ up to 2 secondary/support actions if needed).
- Avoid action labels that are long sentence-like strings.

Link locality rule (strict for downstream UI binding):
- Do not dump unrelated links at the end.
- For each option/row, place the action URL immediately under it.
- If exactly one option has a booking/action URL, keep that action local to that option; do not create a final action-only section for it.
- Use final Quick Actions only for global actions (compare all, official docs, support home, etc.).
- Every action/source link must be an absolute URL with scheme (prefer `https://`).
- Do not output bare domains such as `timeanddate.com` or `www.example.com` without `https://`.

Quality constraints:
- Keep 4-7 headings maximum.
- Prefer concise, information-dense sections over long narrative paragraphs.
- Most sections should use 1-3 short sentences or 1-4 short rows, not dense multi-sentence prose blocks.
- If context notes are needed, keep them to one short sentence; do not add long introductory preambles.
- Avoid large key:value dumps; if many fields exist, summarize them in a table.
- If a URL is provided, it must be a real-world, publicly reachable URL on a real domain.
- Never invent fake domains or placeholder hosts (for instance: static.icons, icon.url, localhost).

Media placement policy (MANDATORY — include media for visual richness):
- Include `Media:` lines to make UI visually rich and app-like.
- Use real, stable image URLs from well-known public sources (Wikimedia Commons, official sites, CDN-hosted assets).
- If a high-confidence image URL is not available for a block, emit icon-only media rather than omitting media entirely.
- Do NOT output standalone `Images:` or `Icons:` sections.
- Place media exactly where it is used in content blocks (option cards, day plans, sections, table rows).
- Never add end-of-response media collections such as "Visual Guide", "Key Feature Icons", "Trip Imagery", "Weather Icons", or "Related Icons". If media is not tied to a specific block, omit it.
- Add media only for blocks where it improves comprehension; avoid decorative or redundant media lines.
- Use this inline format (MANDATORY):
  Media: Image=<url> Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg
- If only one media type is available, include the available type:
  Media: Image=<url>
  Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg
- Keep media local to the related block; do not dump media links at the end.
- Use at most 1 image and 0-1 icon per block to keep UI clean.
- For weather blocks, include a weather icon. For flight blocks, include an airplane icon.
- For playlist/music/entertainment blocks, prefer icon-only media (`music-note`, `music-note-beamed`, or `play-circle`) unless a verified album/cover image URL is available in the current context.
- For travel/place/food blocks, ALWAYS include both image and icon for every block.
- For general/comparison/recommendation blocks, use icon-only media unless a verified, directly relevant image URL is available for the specific option or section.
- For product comparison option rows, use verified/direct option images only when they clearly match the exact product type; otherwise prefer icon-only or no media. Never add a separate comparison gallery.

Asset URL rules (strict — images MUST be content-relevant and working):
- Every Media image MUST visually relate to the content it accompanies. A beach section needs a beach photo, a city section needs a city photo.
- Do not output broken, fake, or placeholder asset URLs (no `<image_url>`, no made-up paths).
- Use real, publicly accessible image URLs that you are confident exist and resolve to actual image files (.jpg, .png, .webp, .svg).
- Prefer images from official sources and stable media hosts with direct file links.
- If a verified image is unavailable for a specific block, use icon-only media for that block; do not force images into general comparison/recommendation answers.
- Do NOT use loremflickr.com, picsum.photos, or other placeholder/random image services.
- For icons, use Bootstrap Icons via jsDelivr CDN (ALWAYS works):
  `https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg`
  Common icon names: `geo-alt`, `calendar`, `clock`, `sun`, `cloud`, `airplane`, `shop`, `star`, `map`, `building`, `cup-hot`, `tree`, `water`, `snow`, `wind`, `thermometer-half`, `currency-rupee`, `ticket-perforated`, `signpost-split`, `cloud-sun`, `moon-stars`, `house`, `car-front`, `phone`, `laptop`, `book`, `music-note`.
- Avoid hosts that are unreliable: images.unsplash.com, cdn.pixabay.com, images.pexels.com, loremflickr.com, picsum.photos.
- Prefer landscape image URLs sized for UI cards (roughly 1200x800 or similar).

Playlist/music response rules:
- Use a short title, one compact mood/context paragraph, and a numbered track table or list.
- Keep track data structured with fields like number, artist, title, and optional mood/genre.
- Do not add standalone media collections or random cyberpunk/album art. If no verified image exists, use only inline Bootstrap music/play icons.

Example of correctly formatted blocks with verified media:
## Taj Mahal, Agra
Media: Image=<verified_image_url_from_context> Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/building.svg
- One of the Seven Wonders of the World, built by Shah Jahan in 1632.
- Best visited at sunrise for the most stunning views.

## Street Food in Bangkok
Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/cup-hot.svg
- Pad Thai and mango sticky rice are must-try dishes.
- Yaowarat Road (Chinatown) has the best night food stalls.

Source rules:
- If verifiable factual claims are included, add 1-3 stable links under Sources.
- For live/local requests, Sources must be included.
- Omit Sources for pure general guidance.
- In Sources, each entry must include a full clickable URL (for example: `https://www.timeanddate.com/weather/`).
- Sources section format is strict:
  - `- <short readable label>: <https://full-url>`
  - never emit bare domains like `timeanddate.com` without scheme
  - never emit source lines that contain only a URL

Output protocol (must follow):
- If given a JSON array of {query_id, query_text}, return a JSON array of {query_id, response_text}.
- If asked for multiple responses to one query, return a JSON array of strings.
- Otherwise, return plain text only.

Now generate the response for user query {query_text}.
