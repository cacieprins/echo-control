# `echo-control` Architectural Handoff: Cooldown Manifold, Antipodal Bail, & Phosphene Visualizer

## 0. Context & Problem Diagnosis

Across a curated $N = 3,680$ track corpus indexed with $d = 512$ CLAP embeddings in Qdrant, greedy `argmax` routing on intro/outro seams + exit velocity produces strong 8-to-10-track local grooves and cross-genre bridges, but suffers from two geometric failure modes on $S^{511}$:

1. **Deterministic Watershed Merging ("Top-100 Radio Rutting"):** Because `argmax` selection is Markovian, the corpus collapses into a functional graph of tributary trees feeding a few canonical trunk sequences. Any two unrelated seeds that touch a shared node merge permanently into the exact same downstream track order across sessions.
2. **CLAP Cone Anisotropy & Hubness:** Pretrained contrastive audio vectors pack into a narrow positive cone around a non-zero corpus mean $\boldsymbol{\mu}$. Tracks with dense broadband mastering sit near subgenre centroids and act as universal nearest-neighbor "hubs," capturing lukewarm transitions by default and pulling trajectories downhill along local density gradients $\nabla \rho(\mathbf{x})$.

The goal is **ergodic groove routing**: preserve smooth 8-to-10-song local trajectories while eroding canonical ruts across sessions, neutralizing broadband hub bias, providing an instant antipodal channel-flip escape (`[b]`), and rendering the live 512D neighborhood as a closed-eye hypnagogic phosphene field.

---

## 1. The Cooldown Manifold & Temperature Routing

### 1.1 Static Startup Precompute (Corpus Centering & CSLS Hubness Tax)
Pull the $N \times 512$ track embeddings once on startup via Qdrant `scroll` ($\approx 7.5\text{ MB}$ in float32; $< 50\text{ ms}$ in NumPy) and compute two static arrays:

1. **Global Corpus Mean & Centered Unit Vectors:**
   $$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i, \qquad \tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}}{\|\mathbf{x}_i - \boldsymbol{\mu}\|_2}$$
    Record the median centered norm $r = \operatorname{median}_i \|\mathbf{x}_i - \boldsymbol{\mu}\|_2$ for mapping centered probes back into raw Qdrant query coordinates.

2. **Cross-Domain Similarity Local Scaling (CSLS) Hubness Penalty ($\bar{s}_j$):**
   For each track $j$'s intro vector $\mathbf{x}_{j,\text{intro}}$, compute its mean cosine similarity to its $m = 20$ nearest neighbors in the library:
   $$\bar{s}_j = \frac{1}{m} \sum_{l \in \mathcal{N}_m(j)} \mathbf{x}_{j,\text{intro}}^\top \mathbf{x}_{l,\text{intro}}$$
   Subtracting $\gamma \bar{s}_j$ ($\gamma \approx 0.30\text{--}0.50$) permanently removes the geometric subsidy enjoyed by broadband "station manager" hubs so specialized interstitial bridges compete on equal footing.

---

### 1.2 Dynamic Spatial Repulsion (Dual-Scale von Mises–Fisher Heat Field)
Treat recent playback history as a repulsive scalar potential $T(\mathbf{x})$ on $S^{511}$ evaluated at candidate centroid $\mathbf{x}_j$:

$$T(\mathbf{x}_j) = \sum_{k \in \mathcal{H}} w_k(t) \exp\!\left(-\frac{1 - \mathbf{x}_j^\top \mathbf{u}_k}{\sigma_{\text{micro}}^2}\right) + \beta \exp\!\left(-\frac{1 - \mathbf{x}_j^\top \bar{\mathbf{c}}_t}{\sigma_{\text{macro}}^2}\right)$$

* **Micro Kernel ($\sigma_{\text{micro}} \approx 0.08\text{--}0.12$, track-count decay $\tau_{\text{micro}} \approx 15$ tracks):** Suppresses exact sonic pockets, alternate radio/extended mixes, and split-alias folder duplicates (e.g., Kanji vs. Romaji tags, alias side projects) purely by geometry without string matching.
* **Macro Basin Kernel ($\sigma_{\text{macro}} \approx 0.25\text{--}0.35$, weight $\beta \approx 0.15\text{--}0.25$):** Driven by a slow exponential moving average of visited centroids:
  $$\bar{\mathbf{c}}_t = \frac{\alpha \bar{\mathbf{c}}_{t-1} + (1 - \alpha)\mathbf{x}_t}{\|\alpha \bar{\mathbf{c}}_{t-1} + (1 - \alpha)\mathbf{x}_t\|_2}, \qquad \alpha \approx 0.80$$
  Spending 4–6 tracks inside a single macro-cluster warms the floor of the entire subgenre crater until peripheral bridge tracks on the rim outscore hubs at the bottom.

---

### 1.3 Persistent Anti-Hebbian Trail Log (`trails.json`)
To prevent fresh sessions from re-entering yesterday's canonical riverbeds, persist sparse timestamps/weights in `~/.local/state/echo-control/trails.json`:

* **Directed Edge Wear ($W_{i \to j}$):** Increment $W_{i \to j} \mathrel{+}= 0.25$ when track $i$ transitions into track $j$, decaying exponentially with wall-clock half-life $t_{1/2,\text{edge}} \approx 14\text{ days}$. Erodes specific `A -> B` ruts after a single listen while allowing track $j$ to be entered freely from other tributaries.
* **Node Play Heat ($H_j$):** Increment $H_j \mathrel{+}= 0.15$ when track $j$ is played, decaying with $t_{1/2,\text{node}} \approx 48\text{--}72\text{ hours}$.
* **Listen-Time Commit Gate:** Only commit $H_j$ and $W_{i \to j}$ to disk/session history once a track has played for $> 30\text{ s}$ or $> 40\%$ of its duration. Rapidly tapping `[n]` to preview/fast-forward downstream trajectories will not scorch unplayed tracks.

---

### 1.4 Candidate Funnel & Close-Call Boltzmann Sampling
1. **Widen Qdrant Retrieval:** Always fetch `limit = 128` (or `256`; exact scan over $3,680$ vectors takes $< 1\text{ ms}$) before applying thermal penalties so boundary bridges outside a dense cluster's raw top-16 remain in the candidate pool.
2. **Adjusted Seam Score:**
   $$\operatorname{Score}'(i \to j) = \operatorname{Seam}(i \to j) - \gamma \bar{s}_j - \lambda_{\text{heat}} T(\mathbf{x}_j) - H_j(t) - W_{i \to j}(t)$$
3. **Masked Top-$k$ Softmax:** Let $S^* = \max_j \operatorname{Score}'(i \to j)$. Retain up to $k = 6$ candidates satisfying $\operatorname{Score}'(i \to j) \ge S^* - \delta$ (margin $\delta \approx 0.06$) and sample with low temperature $\tau \approx 0.02$:
   $$P(j \mid i) = \frac{\exp\!\big(\operatorname{Score}'(i \to j) / \tau\big)}{\sum_{m \in \mathcal{C}_{\delta}} \exp\!\big(\operatorname{Score}'(i \to m) / \tau\big)}$$
   Unmistakable outlier bridges ($> 0.06$ ahead of runner-up) still win deterministically, while close ties inside dense basins naturally branch across different tributaries each day.

---

## 2. Physical Deck Controls: `[n]` Advance vs. `[b]` Channel-Flip

| Key | Action | Audio Behavior | Routing Behavior |
| :--- | :--- | :--- | :--- |
| **`[n]`** | **Track Skip / Advance** | Standard transition | Steps forward along the current track's **outro seam** vector, preserving the planned station groove. |
| **`[b]`** | **Bail / Channel-Flip** | **0 ms Hard Cut** (`replace`) | Jumps $180^\circ$ across the centered manifold to the antipodal sonic continent, flushes the queue, and seeds a fresh groove. |

### 2.1 Why Centered Reflection Beats Raw $-\mathbf{x}$
Because all CLAP vectors lie in a narrow cone around $\boldsymbol{\mu}$, raw negation ($-\mathbf{x}$) points into empty space behind the cone and repeatedly snaps onto the same 3–4 non-musical hull outliers (skits, static, silence). Reflecting **across the corpus centroid $\boldsymbol{\mu}$** keeps the probe dead-center inside the hyperplane of actual music on the opposite wall of the collection.

### 2.2 Multi-Press Bail Stack (Preventing $180^\circ$ Ping-Pong)
If `[b]` simply negates the current track, pressing `[b]` twice in a row (e.g., bailing from Grunge to Ambient, then immediately rejecting Ambient) reflects right back into Grunge. Maintain a short decaying session buffer $\mathcal{B} = \{(\tilde{\mathbf{b}}_1, w_1), (\tilde{\mathbf{b}}_2, w_2), \dots\}$ (decaying over $\approx 3\text{--}4$ bails):

1. **Snapshot & Repel:** Capture active centered vector $\tilde{\mathbf{x}}_{\text{now}}$ (live playhead slice if available, otherwise track centroid), push onto $\mathcal{B}$, and inject a repulsive heat spike into $T(\mathbf{x})$ at $\mathbf{x}_{\text{now}}$ even though listen time was $< 30\text{ s}$.
2. **Compute Antipodal Direction:**
   $$\tilde{\mathbf{q}}_{\text{bail}} = -\frac{\sum_{(\tilde{\mathbf{b}}_m, w_m) \in \mathcal{B}} w_m \tilde{\mathbf{b}}_m}{\left\|\sum_{(\tilde{\mathbf{b}}_m, w_m) \in \mathcal{B}} w_m \tilde{\mathbf{b}}_m\right\|_2}$$
   * *1st press:* Jumps $180^\circ$ to the exact opposite aesthetic.
   * *Immediate 2nd press:* Sum of opposing poles points toward the orthogonal equator between them, visiting a third unrelated genre continent.
3. **Query & Hard-Cut Execution:**
   * Score directly in RAM via $\arg\max_j \big(\tilde{\mathbf{x}}_j^\top \tilde{\mathbf{q}}_{\text{bail}} - H_j(t)\big)$, or map back to raw Qdrant coordinates:
     $$\mathbf{q}_{\text{raw}} = \frac{\boldsymbol{\mu} + r \tilde{\mathbf{q}}_{\text{bail}}}{\|\boldsymbol{\mu} + r \tilde{\mathbf{q}}_{\text{bail}}\|_2}$$
     and pull `limit = 32` to select the top unplayed track.
   * Issue `{"command": ["loadfile", "<path>", "replace"]}` over the `mpv` IPC socket for an instantaneous tuner cut.
   * Clear downstream lookahead queue, plant the selected track at `0.00 / 0.00` (updating `--orbit` origin if active), and compute the new downstream trajectory.

---

## 3. Hypnagogic Phosphene Live Latent Visualizer

Designed for borderless Wayland/QD-OLED output alongside the TUI: renders the active 512D neighborhood at $60\text{ fps}$ as drifting, bioluminescent retinal afterimages over true zero-nit black (`#000000` / `#0b0c10`).

### 3.1 Continuous Playhead Spline $\boldsymbol{\gamma}(t)$
* Poll `time-pos` from `mpv` IPC at $60\text{ Hz}$.
* Given strided slice vectors $\{\tilde{\mathbf{s}}_0, \tilde{\mathbf{s}}_1, \dots, \tilde{\mathbf{s}}_M\} \subset S^{511}$ across the active song (falling back to `{intro, centroid, outro}` 3-point arc if dense slices aren't loaded), evaluate a **spherical Catmull-Rom / cubic slerp spline** $\mathbf{p}(t) = \tilde{\boldsymbol{\gamma}}(t)$ so the camera glides continuously with zero keyframe stepping.

### 3.2 Moving Stiefel Camera $E_t \in V_3(\mathbb{R}^{512})$ with Parallel Transport
Instead of a static global UMAP/PCA hairball, build a local orthonormal 3D tangent frame $E_t = [\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3]$ riding on $\mathbf{p}(t)$:

1. **Forward Axis $\mathbf{e}_1$ (Smoothed Velocity):**
   Take tangent velocity $\mathbf{v}_t = \dot{\mathbf{p}}(t) - (\dot{\mathbf{p}}(t)^\top \mathbf{p}(t))\mathbf{p}(t)$, smooth via EMA ($\alpha_v \approx 0.03$), and normalize to $\mathbf{e}_1$.
2. **Lateral Axis $\mathbf{e}_2$ (Curvature / Seam Pull):**
   Blend instantaneous acceleration $\ddot{\mathbf{p}}(t)$ with the unit vector pointing toward the queued `Next` track's intro seam, project orthogonal to $\operatorname{span}\{\mathbf{p}(t), \mathbf{e}_1\}$, smooth lightly, and normalize to $\mathbf{e}_2$.
3. **Roll-Free Up Axis $\mathbf{e}_3$ (Discrete Parallel Transport):**
   Project the *previous frame's* $\mathbf{e}_{3, t - \Delta t}$ orthogonal to $\operatorname{span}\{\mathbf{p}(t), \mathbf{e}_1, \mathbf{e}_2\}$ and normalize:
   $$\mathbf{e}_{3,t} = \operatorname{normalize}\!\Big(\mathbf{e}_{3,t-\Delta t} - (\mathbf{e}_{3,t-\Delta t}^\top \mathbf{p}_t)\mathbf{p}_t - (\mathbf{e}_{3,t-\Delta t}^\top \mathbf{e}_{1,t})\mathbf{e}_{1,t} - (\mathbf{e}_{3,t-\Delta t}^\top \mathbf{e}_{2,t})\mathbf{e}_{2,t}\Big)$$
   Parallel transport eliminates $180^\circ$ Frenet-frame flips at inflection points so constellations precess smoothly overhead.

### 3.3 Cosine Horizon Fog & Projection
Project all $3,680$ centered tracks in a single `[3680, 512] @ [512, 4]` matrix multiply ($< 0.1\text{ ms}$ on CPU):
* Alignment $a_i = \tilde{\mathbf{x}}_i^\top \mathbf{p}(t)$ and local 3D view coordinates $\mathbf{y}_i = E_t^\top \tilde{\mathbf{x}}_i$.
* **Aperture Visibility Gate:**
  $$\alpha_i = \max\!\left(0, \frac{a_i - a_{\min}}{1 - a_{\min}}\right)^3, \qquad a_{\min} \approx 0.25$$
  Distant genres across the sphere remain completely invisible; only the $\approx 100\text{--}200$ tracks in the local sonic cap bloom softly out of the dark as the song approaches them.

### 3.4 Dual-Decay Ping-Pong Phosphene Shader
Render instanced soft radial Gaussian splats into floating-point ping-pong framebuffers (`FBO_A` $\leftrightarrow$ `FBO_B`) with a fullscreen feedback/advection pass:

* **Curl-Advected Persistence:** Sample the previous frame at `uv + 0.0008 * curlNoise2D(uv * 3.0, time * 0.15)` so trails slowly shear and evaporate like closed-eye retinal mist rather than static lines.
* **Dual Phosphor Half-Lives:**
  * *Fast Core ($\tau_1 \approx 0.4\text{ s}$):* Pale cyan/pearl/violet filament tracing the live playhead ribbon $\mathbf{p}(t)$ and brightening with local spectral flux / velocity norm $\|\dot{\mathbf{p}}(t)\|_2$.
  * *Long Afterimage ($\tau_2 \approx 6.0\text{--}10.0\text{ s}$):* Deep indigo and muted magenta persistence trails decaying asymptotically to zero-nit black.
* **Thermal Manifold Encoding:** Modulate neighbor cloud tint and Gaussian radius by normalized heat $T(\mathbf{x}_i)$:
  * *Cold, unvisited bridges:* Tight, faint bioluminescent blue-green glints that pulse gently when their intro vectors align with $\mathbf{e}_1$.
  * *Hot, exhausted basins:* Wide, diffuse, dim burnt-amber/oxblood embers visibly repelling the trajectory ribbon.
* **`[b]` Channel-Flip Degauss Impulse:** On `[b]` press, do not clear `FBO` to black; trigger a 3-frame radial chromatic-aberration collapse + luminance inversion pulse while re-seeding $E_t$ at the antipode so the old constellation implodes as the new one condenses.

---

## 4. Recommended Implementation Order

1. **Startup Matrix Cache & CSLS (`scorer.py`):** Pull the $3,680 \times 512$ vectors into a local `.npz` / memory cache on startup; compute $\boldsymbol{\mu}$, $r$, $\tilde{\mathbf{X}}$, and $\bar{\mathbf{s}}_{k=20}$.
2. **Trail Persistence & Softmax (`trails.json`):** Add listen-gated $H_j$ and $W_{i \to j}$ state + dual-scale $T(\mathbf{x})$ + `limit = 128` retrieval + $\tau = 0.02$ top-$k$ close-call sampler; optionally expose a `Heat` diagnostic column in the TUI table.
3. **`[b]` Antipodal Cut:** Bind `[b]` in the TUI input loop to maintain the decaying bail stack $\mathcal{B}$, issue `loadfile ... replace` over `/tmp/mpv-socket`, drop a heat spike at the rejected vector, and rebuild the lookahead queue.
4. **Phosphene Window Prototype:** Stand up a standalone Wayland-native `wgpu-py` (or `moderngl` + `glfw`) companion client subscribing to `mpv` IPC + shared trajectory state at $60\text{ fps}$, then port the two WGSL passes (instanced Gaussian splat + curl ping-pong feedback) directly into the Rust `wgpu` binary when unifying `echo-control`.