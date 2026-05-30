## Media and URL rules
- If Assets mapping is provided, use only local media paths from that mapping for Image/Icon. Do not emit remote media URLs for images/icons.
- If an asset's original URL is from random or placeholder hosts such as loremflickr.com, picsum.photos, placehold.co, placeholder, .example, .test, or fake domains, do not use that local path in the IR.
- If no verified image exists, omit the image. Prefer no image over bad or unrelated media; Android can render native placeholders.
- Never invent local paths like /image.jpg or /asset/foo.png.
- SVG/bootstrap/weather icon URLs are Icon only. Never use them as Image URLs, table photo fields, thumbnails, hero images, or entityMedia.image.
- Do not create trailing galleries or icon lists from detached media sections such as Images, Icons, Visual Guide, Trip Imagery, Weather Icons, Related Icons. Attach verified media to the relevant hero/card/table row or drop it.
- Use one hero image only when it adds content value. Use row images only when verified and directly tied to the row.
- Action/source URLs must be public https URLs. Reject http, javascript, data, file, content, intent, localhost, private IPs, .local, .test, .example, malformed hosts, and placeholders.
- Sources with URLs should become compact source/action rows or Buttons with openUrl. Source lines without URLs may be short caption Text.
