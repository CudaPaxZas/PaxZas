# Paxzas CUDA Analyzer (VS Code)

Local **PTX / SASS / CUDA** analysis inside VS Code: no Python, no network. Heuristics for pattern, memory, occupancy, and bottlenecks mirror the parent CudaAnalyzer pipeline in TypeScript.

## Commands

| Command | What it does |
|--------|----------------|
| **Paxzas: Analyze Active CUDA File** | Analyze the open `.ptx`, `.cu`, or `.sass` file. |
| **Paxzas: Analyze Selected File** | From Explorer, analyze the selected artifact. |
| **Paxzas: Analyze CUDA File with Launch Spec** | Optional `threads=…,shared=…,regs=…` overrides. |
| **Run CUDA Kernel Analysis** | Legacy entry; same analysis path. |

Results go to the **CUDA Analyzer** output channel (and a short notification).

## Settings

- **`paxzas.gpuPreset`**: `auto` (PTX arch + `nvidia-smi` for SM count when possible), or `ampere-like-default` / `a100` / `rtx-4090`.

## Development

```bash
npm install
npm run compile   # or npm run watch
npm test
```

**F5** in VS Code with this folder open uses `.vscode/launch.json` (Extension Development Host).

### Package a `.vsix`

```bash
./build-vsix.sh
```

or:

```bash
npm run vsix
```

Install in VS Code: **Extensions → … → Install from VSIX…**

## Requirements

- VS Code **≥ 1.85**
- Optional: **`nvidia-smi`** on PATH (or standard CUDA install paths) for GPU name / `compute_cap` / SM count in `auto` mode
