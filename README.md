# Module 3 Demo: Working with LLM Providers

Twelve small, self-contained Python scripts demonstrating how to call
different LLM providers (OpenAI, Anthropic/Claude, Google Gemini, and local
models via Ollama), and how to reason about multimodal input, sampling
parameters, tool calling, and cost/latency.

Each script is standalone - run any one of them independently.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. Create a `.env` file **one level up** (in the `GenAI` folder, shared
   across modules) with your API keys - `python-dotenv`'s `load_dotenv()`
   searches parent directories automatically, so it will be picked up from
   there:

   - `OPENAI_API_KEY` - from https://platform.openai.com/api-keys
   - `ANTHROPIC_API_KEY` - from https://console.anthropic.com/settings/keys
   - `GOOGLE_API_KEY` - from https://aistudio.google.com/apikey

3. For the Ollama demos (5 and 7), install [Ollama](https://ollama.com)
   locally and pull the models used in this repo:

   ```bash
   ollama pull llama3.2
   ollama pull qwen2.5
   ```

4. For demo 4 (multimodal), place a photo containing one or more animals at
   `assets/animals.png`.

## Demos

| # | File | Provider | What it shows |
|---|------|----------|----------------|
| 1 | `01_openai_basic_example.py` | OpenAI | Minimal chat completion call |
| 2 | `02_anthropic_basic_example.py` | Anthropic | Minimal Claude Messages API call |
| 3 | `03_gemini_basic_example.py` | Gemini | Minimal `generate_content` call |
| 4 | `04_multimodal_image_example.py` | OpenAI | Vision: identifying animals in `assets/animals.png` |
| 5 | `05_ollama_basic_example.py` | Ollama (Llama 3.2) | Running a model fully locally |
| 6 | `06_openai_text_summarization.py` | OpenAI | Summarizing `assets/sample_document.txt` |
| 7 | `07_ollama_qa_example.py` | Ollama (Qwen 2.5) | Q&A grounded in a local document |
| 8 | `08_gemini_multilingual_translation.py` | Gemini | Translating one sentence into 4 languages |
| 9 | `09_anthropic_code_generation.py` | Anthropic | Generating and saving a Python function |
| 10 | `10_sampling_params_demo_gemini.py` | Gemini | Effect of `temperature`, `top_p`, `top_k`, `max_output_tokens` |
| 11 | `11_openai_tool_calling.py` | OpenAI | Function/tool calling with a local weather function |
| 12 | `12_cost_latency_calculator_openai.py` | OpenAI | Approximate per-call cost (gpt-4o-mini pricing) and latency |

## Running a demo

```bash
python 01_openai_basic_example.py
```

## Notes

- **Pricing** in demo 12 is illustrative and may drift - check
  https://openai.com/api/pricing before relying on it for real budgeting.
- Demo 10 uses Gemini specifically because its `GenerationConfig` exposes all
  four sampling parameters (`temperature`, `top_p`, `top_k`,
  `max_output_tokens`) in one place, which makes side-by-side comparison easy.
- Demo 4 requires you to supply your own photo at `assets/animals.png`
  (not included in this repo) - it exits with a clear message if the file
  is missing.
- `assets/generated_code.py` (produced by demo 9) is a generated artifact -
  safe to delete and regenerate by re-running that script.

## Author

**Atul Kahate**
LinkedIn: https://www.linkedin.com/in/atulkahate/
