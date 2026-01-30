# A2UI Android Demo

This sample mirrors the browser-based A2UI shell: enter a prompt, send it to an A2A agent, and render the A2UI response on-device.

## Requirements

- Android Studio (or Gradle CLI)
- A running A2A agent (e.g. restaurant finder) from this repo

## Run

1. Start the agent backend (see `docs/quickstart.md`).
2. Open `samples/client/android` in Android Studio and run the app.
3. Set the server URL:
   - Emulator: `http://10.0.2.2:10002`
   - Device: `http://<your-host-ip>:10002`
4. Try a prompt like `Book a table for 2`.

## Local JSON

You can paste A2UI JSON (single message, array, or JSONL) into the prompt box. If the input parses as JSON, the app renders it locally without calling the server.

## Gemini Direct (No Local Server)

Select **Gemini Direct** and enter your Gemini API key to call Gemini from the app. The default model is `gemini-2.5-flash-lite`. This bypasses the local A2A server and uses the on-device client to request A2UI JSON directly.

## Supported Components

- Text
- Image
- Icon
- Row / Column
- List
- Card
- Button
- TextField
- DateTimeInput
- CheckBox
- Divider
- Modal

## Notes

- The app calls the A2A HTTP+JSON endpoint `POST /message:send` and includes the A2UI extension header.
- Action events from UI components are sent back as `userAction` messages when a server URL is configured.
