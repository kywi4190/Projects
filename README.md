# Kyle Wilson

**Computer Science Major · Applied Mathematics Minor · University of Colorado Boulder** · *Expected graduation: Spring 2027*

I'm working toward empirical AI safety research engineering, with a focus on evaluating, monitoring and controlling LLM agents. My recent work is on multi-agent LLM systems with human oversight, adversarial checks and evaluation harnesses built in. I've also implemented core deep learning and RL algorithms from scratch.

[![Email](https://img.shields.io/badge/Email-kywi4190%40colorado.edu-c41e3a?style=flat-square&logo=gmail&logoColor=white)](mailto:kywi4190@colorado.edu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kyle_Wilson-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kyle-michael-wilson)
[![Resume](https://img.shields.io/badge/Resume-PDF-555555?style=flat-square&logo=readthedocs&logoColor=white)](Kyle_Wilson_Resume_2026.pdf)

---

## Projects at a Glance

| Project | Focus | Stack |
|---|---|---|
| [Deeper Research](#deeper-research-gated-multi-agent-decision-pipeline) | Multi-agent LLM systems, human oversight, evaluation | Python, Claude Agent SDK |
| [Financial RAG Agent](#financial-document-rag-agent-with-agentic-analysis) | RAG, grounding and citation checks, evaluation | Python, LlamaIndex, OpenAI API |
| [Math Visualization Teacher](#prompt-to-video-math-visualization-teacher-with-agentic-ai) | Agentic code generation and self-repair | Python, OpenAI API, Manim |
| [PPO Bipedal Walker](#bipedal-ragdoll-locomotion-with-ppo-from-scratch) | RL from scratch, reward shaping | Python, PyTorch |
| [Drought Prediction](#drought-prediction-from-beetle-specimen-images-with-frozen-vision-backbones) | Uncertainty quantification, generalization to unseen sites | Python, PyTorch |
| [Backprop from Scratch](#handwritten-digit-classification-with-backpropagation-from-scratch) | Neural network fundamentals | C#, Unity |
| [Volumetric Clouds](#volumetric-cloud-simulation-with-compute-shaders) | GPU compute, rendering | C#, HLSL, Unity |
| [L-System Trees](#procedural-tree-generation-with-l-systems) | Procedural generation | C#, Unity |
| [Procedural Terrain](#procedural-terrain-with-mesh-generation) | Noise, mesh generation | C#, Unity |
| [Cellular Automata](#game-of-life-and-temperature-diffusion-with-cellular-automata) | Iterative solvers, emergent behavior | C#, Unity |

---

## LLM Agents & Evaluation

### Deeper Research: Gated Multi-Agent Decision Pipeline
**Focus:** Multi-Agent LLM Systems · Agentic Orchestration · Automated Research · Decision Analysis · LLM Evaluation<br>
**Stack:** `Python` `Claude Agent SDK` `Pydantic` `FastAPI` `HTMX`

- Turns an open-ended goal into a cited decision report in nine stages. The stages map the solution space, scout options, build a scoring rubric, screen, deep-dive, run an adversarial tournament and synthesize the result.
- **Code controls the run and the models only write content.** A deterministic Python orchestrator handles sequencing, budgets, stop rules and human review gates. 20 specialized agent roles produce the content, and each one must output to a strict schema.
- **Every run can be audited.** The run workspace is plain files under git, and every stage and gate decision is a commit. Any run can be inspected, diffed, resumed after a crash or partially rerun.
- **Humans review at three points.** Three review gates, including the angle map and rubric, accept typed edits through YAML or a browser viewer. Both paths use the same validation code.
- **Checks against bias and weak evidence:**
  - Tool hooks hide user preferences from most agents.
  - Destination-only and preference-adjusted scoreboards are compared to catch rank flips.
  - An independent verifier re-checks key claims.
  - Prosecutor and steelman agents argue against the leading options.
- **Budgets are computed in plain code:**
  - Prior-weighted allocation with exploration floors.
  - A saturation rule that stops angle mapping.
  - A convergence rule that ends the tournament.
  - A hard budget cap checked before every agent call.
- **Evaluation harness:** scores any run against seeded benchmarks on breadth, depth, informedness, quality and anti-overfitting, using a Haiku-class judge.
- Covered by 1,000+ pytest tests, including a full offline mock run from start to finished report. Live runs have been used on real research questions.

[**View Repository →**](https://github.com/kywi4190/deeper-research)

---

### Financial Document RAG Agent with Agentic Analysis
**Focus:** Agentic AI · Retrieval-Augmented Generation · LLMs · NLP · Finance<br>
**Stack:** `Python` `LlamaIndex` `ChromaDB` `Streamlit` `OpenAI API`

- Streamlit app that pulls SEC 10-K and 10-Q filings from EDGAR and answers questions about them with a citation for every answer. It also writes investment memos from the filings.
- Hybrid search merges BM25 keyword search and dense vector search with Reciprocal Rank Fusion, then reranks the results with a cross-encoder.
- Chunking follows each filing's structure. Sections stay intact, financial tables are never split, and every chunk is tagged with ticker, year and section.
- A corrective RAG loop scores how relevant the retrieved text is and rewrites the query when confidence is low. It then checks that the answer is supported by the sources.
- Three agents write each investment memo: financial data, qualitative analysis and synthesis. Numbers come from XBRL facts, and ratios such as margins and debt-to-equity are calculated from them.
- An evaluation harness combines RAGAS metrics with custom citation-accuracy and numerical-accuracy scores. It runs on 50 curated questions about AAPL, MSFT and GOOGL filings, and about 150 pytest tests cover the pipeline.

[**View Repository →**](https://github.com/kywi4190/Financial-Rag-Agent)

---

### Prompt-to-Video Math Visualization Teacher with Agentic AI
**Focus:** Agentic AI · Generative AI · LLMs · Math Education<br>
**Stack:** `Python` `HTML` `OpenAI API` `Manim`

- Locally hosted web app that turns a prompt into a 25–45 second educational video that visualizes and explains a math concept intuitively.
- Renders mathematical animations with Manim and narration with OpenAI TTS, with subtitles overlaid.
- A GPT-5 agent pipeline plans a teaching strategy and writes structured Manim JSON. It critiques and rewrites its own code to reduce errors, and when a render fails it reads the logs and repairs the code.
- Code sanitization and detailed system and user prompts raise the success rate and keep output within the required features.

[**View Repository →**](https://github.com/kywi4190/Prompt-to-Math-Visualization-with-Agentic-AI)

---

## Deep Learning & Reinforcement Learning

### Bipedal Ragdoll Locomotion with PPO from Scratch
**Focus:** Reinforcement Learning · Continuous Control · Reward Shaping · Physics Simulation<br>
**Stack:** `Python` `PyTorch` `Pymunk` `Pygame`

- Implemented Proximal Policy Optimization from scratch to train a 7-segment 2D ragdoll (torso, thighs, shins and feet) to walk using 6 motorized joints.
- The actor and critic are separate MLPs:
  - The policy is Gaussian, with tanh-bounded means and a state-dependent log std.
  - The critic is a value network trained with GAE.
  - Training uses a clipped surrogate loss, an entropy bonus and a cosine-annealed learning rate.
- Custom Gym-style environment built on Pymunk rigid-body physics, with a 19-dim observation space, joint limits, per-joint motor strengths and foot contact sensors.
- Iterative reward shaping removed reward-hacking strategies such as diving forward and standing still. The final reward combines a compounding alive bonus, a superlinear velocity reward, an energy penalty and a dominant fall penalty.
- 5 training profiles scale from 500K to 50M timesteps, with CUDA support, per-component reward diagnostics and automatic saving of the best checkpoint.
- Pygame visualizer with a scrolling camera, a Perlin-noise parallax background and live hot-swapping between trained models.

[**View Repository →**](https://github.com/kywi4190/PPO-Bipedal-Walker)

---

### Drought Prediction from Beetle Specimen Images with Frozen Vision Backbones
**Focus:** Uncertainty Quantification · Generalization to Unseen Sites · Transfer Learning · Computer Vision · Ecology<br>
**Stack:** `Python` `PyTorch` `Optuna` `HuggingFace Transformers` `OpenCLIP`

- Predicts three SPEI drought indices from 22,370 carabid beetle images. The model outputs a mean and standard deviation for each index and is scored by Continuous Ranked Probability Score (CRPS).
- Embeddings from frozen DINOv3 and BioCLIP-2 backbones are extracted once and cached, which makes 200-trial Optuna architecture searches cheap to run.
- A split-horizon ensemble pairs a DINOv3 model for 30-day SPEI with a BioCLIP-2 plus species-embedding model for 1-year and 2-year SPEI.
- Seven pooling strategies combine the variable-size image sets from each trap event. ColorChecker color-correction matrices separate beetle morphology from camera and lighting drift.
- Validated on held-out NEON sites to test generalization to unseen locations. The ensemble reaches **0.775 RMS-CRPS vs. 0.885** for a single-model baseline (lower is better).

[**View Repository →**](https://github.com/kywi4190/SMood-ML-Challenge)

---

### Handwritten Digit Classification with Backpropagation from Scratch
**Focus:** Deep Learning · Supervised Learning · Image Classification<br>
**Stack:** `C#` `Unity`

- Implemented a fully parameterized feedforward neural network and backpropagation without ML libraries, to learn how deep learning works at a low level.
- The training loop includes data shuffling, mini-batch gradient descent, adaptive learning-rate decay and in-editor logging of loss and accuracy.
- Unity workflow with CSV normalization, one-hot encoding, textured digit previews and evaluation metrics for training and testing.

[**View Repository →**](https://github.com/kywi4190/Handwritten-Digit-Classifier)

---

## Graphics & Simulation

### Volumetric Cloud Simulation with Compute Shaders
**Focus:** 3D Graphics · Compute Shaders · Parallelization · Natural Science<br>
**Stack:** `C#` `HLSL` `Unity`

- Real-time volumetric ray-marching renderer that models how light diffuses through clouds.
- Compute shaders run noise generation and density sampling in parallel, fast enough for real-time use in games.
- Procedurally generated 3D Worley noise volumes and 2D Perlin density textures give seeded, infinitely tileable cloud structure.
- Parameters control wind, coverage, density, light scattering and level of detail.

[**View Repository →**](https://github.com/kywi4190/Volumetric-Clouds)

---

### Procedural Tree Generation with L-Systems
**Focus:** 3D Graphics · Fractals · Game Development · Natural Science<br>
**Stack:** `C#` `Shader Graph` `Unity`

- Implemented Lindenmayer-system fractals to procedurally generate different tree species.
- Generation is randomized and highly parameterized, so trees vary widely while staying controllable.
- Interactive demo with real-time generation, species selection and an orbital camera rig.

[**View Repository →**](https://github.com/kywi4190/Procedural-Tree-Generation)

---

### Procedural Terrain with Mesh Generation
**Focus:** 3D Graphics · Game Development · Noise Generation<br>
**Stack:** `C#` `Shader Graph` `Unity`

- Configurable multi-octave Perlin noise generates the mesh heightmap.
- Noise sampling, biome classification, gradient-based coloring and a custom shader produce realistic islands. The physics-ready meshes suit game development or ML training environments.

[**View Repository →**](https://github.com/kywi4190/Procedural-Terrain-Generation)

---

### Game of Life and Temperature Diffusion with Cellular Automata
**Focus:** Iterative Solvers · Cellular Automata · Emergent Behavior<br>
**Stack:** `C#` `Unity`

- Implemented a Gauss-Seidel iterative solver to simulate heat propagation and gas convection.
- The same cellular automata system also simulates the emergent behavior of Conway's Game of Life.
- A 2D texel grid stores cell states, and each cell updates according to its neighbors' states and a set of rules.

[**View Repository →**](https://github.com/kywi4190/Cellular-Automata)
