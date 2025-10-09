# Genkit Flow

## Setup

To use the models, you need to set the following environment variables with your API keys:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

You can set these in a `.env` file in the root of the project, or in your shell's configuration file (e.g., `.bashrc`, `.zshrc`).

You also need to install dependencies before running:
```bash
npm install
```

## Running all evals (warning: can use *lots* of model quota)

To run the flow, use the following command:

```bash
npx tsx src/index.ts
```

## Running a Single Test

You can run the script for a single model and data point by using the `--model` and `--prompt` command-line flags. This is useful for quick tests and debugging.

### Syntax

```bash
npx tsx src/index.ts --model='<model_name>' --prompt=<prompt_name>
```

### Example

To run the test with the `gemini-2.5-flash (thinking: 0)` model and the `dogBreedGenerator` prompt, use the following command:

```bash
npx tsx src/index.ts --model='gemini-2.5-flash' --prompt=dogBreedGenerator
```

## Controlling Output

By default, the script only prints the summary table and any errors that occur during generation. To see the full JSON output for each successful generation, use the `--verbose` flag.

### Example

```bash
npx tsx src/index.ts --verbose
```
