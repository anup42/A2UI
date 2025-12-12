# Angular A2UI

These are sample implementations of A2UI in Angular.

## Prerequisites

1. [nodejs](https://nodejs.org/en)

NOTE: [For the rizzcharts app](../../agent/adk/rizzcharts/), you will need GoogleMap API ([How to get the API key](https://developers.google.com/maps/documentation/javascript/get-api-key)) to display Google Map custome components. Please refer to [Rizzcharts README](./projects/rizzcharts/README.md)

## Running

Here is the quickstart for the restaurant app:

```bash
# Export your Gemini API key
export GEMINI_API_KEY=your_gemini_api_key
echo "export GEMINI_API_KEY=your_gemini_api_key" >> .env
cp .env ../../agent/adk/restaurant_finder/.env

# Start the restaurant app frontend
npm install 
npm run demo:restaurant 
```

Here are the instructions if you want to do each step manually. 

1. Build the shared dependencies by running `npm install && npm run build` in the `renderers/lit` directory
2. Install the dependencies: `npm install`
3. Run the relevant A2A server:
  * [For the restaurant app](../../agent/adk/restaurant_finder/)
  * [For the contact app](../../agent/adk/contact_lookup/)
  * [For the rizzcharts app](../../agent/adk/rizzcharts/)
4. Run the relevant app:
  * `npm start -- restaurant`
  * `npm start -- contact`
  * `npm start -- rizzcharts`
  * `npm start -- gallery` (Client-only, no server required)
5. Open http://localhost:4200/

Here is the end to end commands for the restaurant app:
