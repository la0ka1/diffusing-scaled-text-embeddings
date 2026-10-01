---
title: "Scaling and Distilling Text Embeddings for Better Diffusibility"
permalink: /
layout: single
classes: wide
---

<p class="home-heading"><a href="https://la0ka1.github.io/" aria-label="Back to homepage"><span aria-hidden="true">&larr;</span> Back to homepage</a></p>

<!-- TODO before publishing: set the arXiv link; add alphaXiv / slides buttons when they exist. We can also publish first and then update. -->
<p class="button-row">
<a class="btn btn--success" href="{{ site.github.repository_url }}"><i class="fab fa-github" aria-hidden="true"></i> Code</a>
<a class="btn btn--arxiv" href="#"><i class="fas fa-file-alt" aria-hidden="true"></i> arXiv (coming soon)</a>
<a class="btn btn--hf" href="https://huggingface.co/collections/la0ka1/diffusing-scaled-text-embeddings-6abd7ff3c91fd70bd749c197"><i class="fas fa-cubes" aria-hidden="true"></i> Checkpoints</a>
</p>

<p class="author-row">
<a class="author-link" href="https://la0ka1.github.io/"><strong>Zekai Zhang</strong></a><sup>1</sup>, <a class="author-link" href="https://sunsmarterjie.github.io/">Yunjie Tian</a>, <a class="author-link" href="https://yanjinhe.github.io/">Yanjin He</a><sup>1</sup>, <a class="author-link" href="https://xiaoyanzhang1.github.io/">Xiaoyan Zhang</a><sup>1</sup>, Dongdi Zhao, <a class="author-link" href="https://qingqu.engin.umich.edu/">Qing Qu</a><sup>1</sup>, and Di Fu
</p>
<p class="affiliation-row">
<sup>1</sup>University of Michigan
</p>

<p class="tldr-box"><strong>TL;DR.</strong> Scaling the latent embeddings of continuous diffusion language models (DLMs) improves their performance but is not enough. Distilling them with soft labels further makes them more diffusible.</p>

---

<p class="lead-italic"><em>Which embedding should a continuous diffusion language model use?</em></p>
<p>A continuous diffusion language model (DLM) is latent diffusion on text: a frozen <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-setup" data-note-target="note-setup">embedding model</button> maps tokens to embeddings, a denoiser learns to generate these embeddings, and a decoder maps them back to tokens.</p>
<div id="note-setup" class="inline-note-body" hidden>

<p>We follow <a href="https://github.com/Ugness/ELF-pytorch">ELF</a>, where a token sequence \(\bm{s}\) becomes \(\bm{x}=\mathrm{Emb}(\bm{s})\in\mathbb{R}^{L\times d}\), and the rest is standard Gaussian diffusion:</p>
$$
\mathcal{L} = \mathbb{E}_{t,\bm{x},\bm{\epsilon}}\Big[\tfrac{1}{(1-t)^2}\,\big\|\bm{x}_\theta(\bm{z}_t,t)-\bm{x}\big\|^2\Big], \qquad \bm{z}_t = t\bm{x}+(1-t)\bm{\epsilon}.
$$
<p>Early work tried co-trained or simple embeddings (from a lookup table), but later people found that pretrained embeddings improve over them, and we seek their full potential.</p>

</div>

<p>The image diffusion guys have asked the same question for years: which latent is easy to diffuse? For text it is mostly open. So we keep the diffusion fixed and change only the embedding.</p>

<img class="feature-figure" src="{{ '/assets/figures/fig_teaser_combined.png' | relative_url }}" alt="Left: generative perplexity against entropy for the same diffusion model trained on different embeddings. Right: generated embeddings among candidate words, for T5Gemma-2 and for the distilled student." width="90%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Overview.</strong> Left: the same DLM on different embeddings. Right: on T5Gemma-2 a generated embedding can deviate from the plausible words; on the distilled student it lands among them.</p>

---
<p class="lead-italic"><em>A scaling behavior in the latents for continuous DLMs.</em></p>
<p>We find a <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-scaling" data-note-target="note-scaling">scaling</button> behavior within the T5 family of encoder-decoder models (T5-small/base &rarr; T5Gemma-1 &rarr; T5Gemma-2): generation improves along with the models. T5Gemma-2 cuts the generative perplexity by about 40% at the same entropy as T5-small.</p>
<div id="note-scaling" class="inline-note-body" hidden>

<p>By scaling we mean updating to a stronger embedding model of the same family, pretrained with more data and a better recipe, and not only adding parameters.</p>

</div>

<img class="feature-figure figure--narrow" src="{{ '/assets/figures/fig_panel512.png' | relative_url }}" alt="Generative perplexity against entropy for the same diffusion model trained with different encoders." width="60%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Same DLM, different embedding models.</strong> Lower perplexity at higher entropy is better; the star is real text.</p>

<!-- <p>But a strong encoder is not enough by itself: the encoder-only ModernBERT does well on downstream tasks, but it is less suitable for diffusion.</p> -->

---
<p class="lead-italic"><em>But scaled embeddings are hard to generate perfectly.</em></p>
<p>When we look at the generated embeddings, some of them are far from every word that could fit the position. We call such an embedding <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-invalid" data-note-target="note-invalid">invalid</button>. It can still be decoded, but with low confidence or to a wrong word, and it tends to cause grammar mistakes that hurt generation.</p>
<div id="note-invalid" class="inline-note-body" hidden>

<p>For a position, take the top-\(k\) candidate words of the decoder and their real embeddings \(\bm{x}_1,\dots,\bm{x}_k\). A generated embedding \(\hat{\bm{x}}\) is invalid if</p>
$$
\min_i \|\hat{\bm{x}}-\bm{x}_i\| \;>\; 0.91\,\mathrm{nn},
$$
<p>where nn is the median nearest-neighbor distance between real embeddings.</p>

</div>

<img class="feature-figure" src="{{ '/assets/figures/fig_degen_realvsgen_w2.png' | relative_url }}" alt="A generated sample with uncertain tokens marked, and four positions drawn among their candidate words." width="90%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Embedding errors in a generated sample.</strong> Purple tokens are decoded with low confidence. Below, each disk is the range of a candidate word; an invalid embedding lands outside every disk.</p>

<p>This may be an inherent problem of continuous diffusion trained with MSE. <strong>MSE is not "accurate": it is mode-averaging.</strong> The MSE-optimal denoiser predicts the <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-mse" data-note-target="note-mse">conditional mean</button>: when several words fit a position, it is a combination of them. A scaled encoder keeps these words far apart, making the combination ambiguous and guiding the sampling into the empty space between them.</p>
<div id="note-mse" class="inline-note-body" hidden>

<p>The minimizer of \(\mathbb{E}\,\|\bm{x}_\theta(\bm{z}_t,t)-\bm{x}\|^2\) is \(\bm{x}_\theta(\bm{z}_t,t)=\mathbb{E}[\bm{x}\mid\bm{z}_t]\). If the position could hold the words \(\bm{x}_1,\dots,\bm{x}_k\) with probabilities \(p_1,\dots,p_k\), this is \(\sum_i p_i\bm{x}_i\): a point inside their convex hull.</p>

</div>

---
<p class="lead-italic"><em>Distilling with soft labels reduces the embedding error.</em></p>
<p>We distill the T5Gemma-2 encoder into a student encoder that learns the teacher decoder's probabilities as <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-kd" data-note-target="note-kd">soft labels</button>. The soft labels contain information about the plausible words of a position, so the student places these words closer and becomes more robust.</p>
<div id="note-kd" class="inline-note-body" hidden>

$$
\mathcal{L}_{\mathrm{KD}}(\theta)=\sum_{j=1}^{L}\mathrm{KL}\big(p^{\mathrm{T}}_j\,\|\,p^{\mathrm{S}}_j\big),\qquad
p^{\mathrm{T}}=\mathrm{Dec}^{\mathrm{T}}(\mathrm{Enc}^{\mathrm{T}}(\bm{s})),\quad
p^{\mathrm{S}}=\mathrm{Dec}^{\mathrm{T}}(\mathrm{Enc}^{\mathrm{S}}_\theta(\bm{s})).
$$
<p>The decoder is the frozen teacher decoder in both cases. The student has 9 of the teacher's 18 layers.</p>

</div>

<img class="feature-figure" src="{{ '/assets/figures/fig_distill_unified.png' | relative_url }}" alt="The distillation pipeline and its effect on the embeddings of candidate words." width="80%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Distillation.</strong> The student matches the teacher's decoded probabilities, and the embeddings of plausible words move closer.</p>

<p>As a result, the same DLM trained on the student's embeddings generates fewer invalid embeddings.</p>

<img class="feature-figure" src="{{ '/assets/figures/fig_student_mitigation_combined.png' | relative_url }}" alt="Generated samples and embedding panels for T5Gemma-2 and for the distilled student." width="90%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Teacher against student.</strong> The student's generated embeddings lie in the region formed by the candidate words.</p>

---
<p class="lead-italic"><em>Results.</em></p>
<p>On OpenWebText with 1024-token sequences, the student wins at every sampling budget:</p>

<!-- | Model | Gen. PPL (lower is better) | Entropy |
|---|---|---|
| Real text | 15.4 | 5.43 |
| GPT-2-M (autoregressive) | 20.8 | 5.45 |
| T5Gemma-2 + ELF-M | 19.3 | 5.44 |
| **Student (ours) + ELF-M** | **17.8** | 5.45 | -->

<img class="feature-figure" src="{{ '/assets/figures/fig_sampling_abl_side.png' | relative_url }}" alt="Generative perplexity against entropy over sampling settings, for T5Gemma-2 and the student." width="90%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Over sampling settings.</strong> The student outperforms T5Gemma-2 from 64 to 512 sampling steps, while the teacher collapses at few steps.</p>

<p>The distillation trades discrimination for diffusibility (SST-2 probing drops from 89.4 to 78.1).</p>

---
<p class="lead-italic"><em>Future directions.</em></p>
<ul>
<li>Few-step generation: ELF trained on raw T5Gemma-2 embeddings collapses at small NFEs, and a forgiving latent should hold up better.</li>
<li>Larger models for practical tasks such as QA, as that is where a latent has to earn its keep.</li>
<li>Using autoregressive models directly as embedding models, dropping the Gemma &rarr; T5Gemma &rarr; ELF detour: if an LLM's own hidden states can be made diffusible, continuous DLMs ride on every LLM release.</li>
</ul>

---

### BibTeX

```bibtex
@article{zhang2026scaling,
  title={Scaling and Distilling Text Embeddings for Better Diffusibility},
  author={Zhang, Zekai and Tian, Yunjie and He, Yanjin and Zhang, Xiaoyan and Zhao, Dongdi and Qu, Qing and Fu, Di},
  year={2026}
}
```
