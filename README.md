# sd-forge-rouwei-gemma-adapter

`sd-forge-rouwei-gemma-adapter` is a reimplementation of the [ComfyUI LLM SDXL Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter) node pack by [NeuroSenko](https://github.com/NeuroSenko) for [stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge).

It is built for using [Rouwei-Gemma](https://civitai.com/models/1782437/rouwei-gemma), an adapter release by [MinthyBasis](https://github.com/MinthyBasis) trained to use Google's Gemma and T5Gemma models as text encoders for SDXL, inside ReForge.

If you want Rouwei-Gemma in ReForge, that is what this extension is for. ReForge does not make this especially elegant, so the extension handles the annoying part for you: host `transformers`, the launch flag nonsense, and the mildly cursed amount of monkey patching needed to make this work at all.

It may also be compatible with [other](https://github.com/lllyasviel/stable-diffusion-webui-forge) [WebUI](https://github.com/automatic1111/stable-diffusion-webui) implementations, but it has not been tested with those hosts.

## Start Here

Start with the [Rouwei-Gemma release on Civitai](https://civitai.com/models/1782437/rouwei-gemma).

The current recommended path is the `v0.2_t5gemma2b` adapter from that release. That is the setup this README is centered around.

## What You Need

- ReForge
- this extension installed in `extensions\`
- a Rouwei-Gemma adapter from Civitai
- a compatible encoder model
- an SDXL checkpoint

Recommended encoder download:

- the official [T5-Gemma 2B-2B UL2 model page](https://huggingface.co/google/t5gemma-2b-2b-ul2)

Start with RouWei if you want the closest match to the published release. Once that is working, try the same setup with other Illustrious-based checkpoints and merges.

## Supported Setups

- recommended: RouWei `v0.2_t5gemma2b` adapter with the `t5gemma` preset
- legacy: older Gemma-family RouWei adapters with the `gemma` preset
- adapter weights stored as `.safetensors`

## Install

Install the extension with either of these paths:

1. Use the ReForge GitHub extension installer.
2. Clone this repository into `extensions\`.

The extension installs its own lightweight requirements from `requirements.txt`:

- `safetensors>=0.5.2`
- `sentencepiece>=0.2.0`

It also handles the part ReForge makes stupid:

- upgrades host `transformers` to `4.53.1`
- generates `webui-user-gemma-adapter.bat` in the ReForge root

## Launch ReForge

Launch ReForge with `webui-user-gemma-adapter.bat`.

That is the supported startup path for this extension. The managed launcher:

- keeps your current `COMMANDLINE_ARGS` from `webui-user.bat`
- adds `--skip-install`
- checks the ReForge venv `transformers` version on every launch
- repairs `transformers` automatically when it drifts below `4.53.1`

Use `webui-user.bat` to manage your normal launch args. Use `webui-user-gemma-adapter.bat` when you actually want Rouwei-Gemma to keep working.

Development and validation used `transformers 4.53.1`.

## Download the Adapter

Download the adapter from [Rouwei-Gemma on Civitai](https://civitai.com/models/1782437/rouwei-gemma).

Place the adapter `.safetensors` file in:

- `models\llm_adapters\`

For the current recommended setup, use the `v0.2_t5gemma2b` adapter from that release.

## Download the Encoder Model

For the current recommended setup, download [google/t5gemma-2b-2b-ul2 on Hugging Face](https://huggingface.co/google/t5gemma-2b-2b-ul2).

Accept the Gemma license terms on Hugging Face before downloading the files.

Place the full model directory under:

- `models\llm\<model-directory>`

The extension expects a normal Hugging Face model folder with the config, tokenizer, and model files kept together in one directory.

If you are following an older Gemma-family RouWei adapter setup instead, place that Gemma-family model directory under the same `models\llm\` root and use the `gemma` preset in ReForge.

## Model and Adapter Files

The extension scans these locations under the active ReForge models root:

- `models\llm\<model-directory>`
- `models\llm_adapters\<adapter>.safetensors`

Example layout:

```text
stable-diffusion-webui-reForge\
\models\
|-- llm\
|   \-- t5gemma-2b-2b-ul2\
|       |-- config.json
|       |-- generation_config.json
|       |-- model.safetensors
|       |-- tokenizer.json
|       |-- tokenizer.model
|       \-- tokenizer_config.json
\-- llm_adapters\
    \-- rouweiT5Gemma_v02.safetensors
```

Each model choice must be a complete Hugging Face directory with config and tokenizer files. Adapter choices are standalone `.safetensors` files.

## ReForge UI

The extension appears as an always-visible block named `RouWei LLM Adapter`.

Shared controls:

- `Enable`
- `LLM model`
- `Adapter`
- `Preset`
- `Force reload`
- `Device`

Gemma controls:

- `System prompt (Gemma)`
- `Skip first tokens (Gemma)`

T5-Gemma controls:

- `Max length (T5-Gemma)`

## Usage

1. Start ReForge with `webui-user-gemma-adapter.bat`.
2. Load an SDXL checkpoint.
3. Open `RouWei LLM Adapter`.
4. Enable the extension.
5. Select the downloaded encoder model directory.
6. Select the adapter file.
7. Set `Preset` to match the downloaded adapter.
8. Start with the default preset values:
   - `t5gemma`: `Max length = 512`
   - `gemma`: default system prompt and `Skip first tokens = 27`
9. Generate normally.

Use `Force reload` when you change model files on disk or want to rebuild the loaded model and adapter state.

For the current Civitai release, the usual choice is:

- the `v0.2_t5gemma2b` adapter
- a T5-Gemma encoder model
- the `t5gemma` preset

Once the primary RouWei setup is working, try it with other Illustrious-based SDXL checkpoints. The published RouWei release notes specifically call out broad compatibility with Illustrious-family models, including popular merges.

## Generation Metadata

When enabled, the extension records:

- preset
- model family
- selected model
- selected adapter
- Gemma settings when the `gemma` preset is active
- T5-Gemma `max_length` when the `t5gemma` preset is active

## Troubleshooting

If the model or adapter dropdown is empty:

- confirm the files are under `models\llm\` and `models\llm_adapters\`
- restart ReForge after adding new model or adapter files
- confirm ReForge is using the models root you expect

If generation fails as soon as the extension is enabled:

- confirm the active checkpoint is SDXL
- confirm the selected model directory includes config and tokenizer files
- confirm the selected adapter matches the chosen preset
- confirm the current RouWei `v0.2_t5gemma2b` adapter is paired with the `t5gemma` preset

If Gemma-family loading fails because the architecture is not recognized:

- start ReForge with `webui-user-gemma-adapter.bat`
- let the managed launcher repair `transformers` before startup

If the `t5gemma` preset fails because `T5GemmaEncoderModel` is unavailable:

- start ReForge with `webui-user-gemma-adapter.bat`
- confirm the managed launcher completed the `transformers` upgrade step

If ReForge checkpoint loading fails on `no_init_weights`:

- use a host `transformers` build that includes `no_init_weights`, or
- use the compatibility shim path already bundled with this extension

If Gemma or T5-Gemma support disappears after a later restart:

- start ReForge with `webui-user-gemma-adapter.bat`
- let the managed launcher refresh `transformers` before the UI loads

## Attribution

This extension reimplements the RouWei Gemma and T5-Gemma path from [ComfyUI LLM SDXL Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter) by [NeuroSenko](https://github.com/NeuroSenko) for ReForge.

The upstream project is MIT-licensed, and this repository preserves that attribution.

## License

This project is licensed under MIT. I usually license my own work under GPL, but this one stays MIT out of respect for NeuroSenko's original implementation and the licensing choice it was released under.

See [LICENSE](LICENSE).

## From the Developer 💖

I ported this for a friend, and I do not really use ReForge myself.

Still, if you want Rouwei-Gemma in ReForge without manually babysitting `transformers`, launch flags, and all the other little bits of nonsense, that is exactly what this extension is for.

- **Buy Me a Coffee**: You can help fuel more projects like this at my [Ko-fi page](https://ko-fi.com/artificial_sweetener).
- **My Website & Socials**: See my art, poetry, and other dev updates at [artificialsweetener.ai](https://artificialsweetener.ai).
- **If you like this project**, it would mean a lot to me if you gave me a star here on Github!! ⭐
