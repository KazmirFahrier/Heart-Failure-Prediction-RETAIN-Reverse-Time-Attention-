<h1>Heart Failure Prediction — RETAIN (Reverse Time Attention) • HW4</h1>

<p>
This project implements <strong>RETAIN</strong>, a reverse-time attention RNN for interpretable clinical risk prediction.
We train on longitudinal diagnosis sequences (ICD-9 label indices) and predict Heart Failure (HF) risk, while
surfacing visit-level (<em>α</em>) and code-level (<em>β</em>) importance.
</p>

<hr/>

<h2>📁 Project Layout</h2>
<pre><code>.
├─ HW4_RETAIN.ipynb
├─ HW4_RETAIN-lib/
│  └─ data/
│     └─ train/
│        ├─ pids.pkl   # patient ids
│        ├─ vids.pkl   # visit ids per patient
│        ├─ hfs.pkl    # labels (0=non-HF, 1=HF)
│        ├─ seqs.pkl   # diagnosis label sequences: list[list[list[int]]]
│        ├─ types.pkl  # label → string map size 619
│        └─ rtypes.pkl # reverse map
└─ README.md
</code></pre>

<hr/>

<h2>🧠 Problem &amp; Data</h2>
<ul>
  <li><strong>Goal:</strong> Predict HF from a patient’s sequence of diagnosis labels across visits.</li>
  <li><strong>Dataset:</strong> 1,000 patients (train/val split inside the notebook); label space = <code>619</code>.</li>
  <li><strong>Class balance (train):</strong> HF positives = <code>548</code> (~<code>0.55</code> ratio).</li>
  <li><strong>Sequence schema:</strong> <code>seqs[i][j][k]</code> = k-th diagnosis label at the j-th visit for the i-th patient.</li>
</ul>

<hr/>

<h2>🚦 Pipeline</h2>

<h3>1) Dataset &amp; Collate (padding, masks, reverse)</h3>
<ol>
  <li><strong>CustomDataset</strong>: stores raw Python objects <code>(sequences, labels)</code>.</li>
  <li><strong>collate_fn</strong>:
    <ul>
      <li>Pad to common shape <code>(B, Vmax, Cmax)</code> → <code>x: Long</code>, <code>masks: Bool</code>.</li>
      <li>Create time-reversed tensors <code>rev_x</code>, <code>rev_masks</code> by flipping only the <em>true</em> visits.</li>
      <li>Return <code>(x, masks, rev_x, rev_masks, y)</code> with <code>y: Float</code> and shape <code>(B,)</code>.</li>
    </ul>
  </li>
  <li><strong>DataLoader</strong>: signature <code>load_data(train_dataset, val_dataset, collate_fn)</code>;
      <code>batch_size=32</code>; shuffle only the train loader. With the provided split, <code>len(train_loader)=25</code>.</li>
</ol>

<h3>2) RETAIN architecture</h3>
<pre><code>Embedding(num_codes=619, emb_dim=128)
RNN-α: GRU(input=128, hidden=128, batch_first=True)
RNN-β: GRU(input=128, hidden=128, batch_first=True)
AlphaAttention: Linear(128 → 1) + softmax over time    # visit importance α (B,V,1)
BetaAttention : Linear(128 → 128) + tanh               # code-group importance β (B,V,128)
Context c     : Σⱼ αⱼ · (βⱼ ⊙ vⱼ)                      # vⱼ = visit representation (sum of code embeddings)
Head          : Linear(128 → 1) + Sigmoid              # probability p(y=1)
</code></pre>

<p>
<strong>Forward (reverse time):</strong> embed reversed visits → sum code embeddings per visit with masks →
run GRU-α and GRU-β → compute α (softmax over visits) and β (tanh) →
masked, weighted sum to get context <code>c</code> → linear + sigmoid → <code>p(HF)</code>.
</p>

<h3>3) Training &amp; Evaluation</h3>
<ul>
  <li><strong>Loss:</strong> Binary Cross-Entropy (<code>BCELoss</code>) on probabilities.</li>
  <li><strong>Optimizer:</strong> Adam (<code>lr=1e-3</code>).</li>
  <li><strong>Metrics:</strong> Precision, Recall, F1 (threshold 0.5) and ROC-AUC (using probabilities).</li>
  <li><strong>Epochs:</strong> 5 (default run in the notebook).</li>
</ul>

<hr/>

<h2>📈 Observed Results (Validation)</h2>
<table>
  <thead>
    <tr><th>Epoch</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>0.75</td><td>0.83</td><td>0.78</td><td>0.83</td></tr>
    <tr><td>2</td><td>0.77</td><td>0.74</td><td>0.75</td><td>0.84</td></tr>
    <tr><td>3</td><td>0.78</td><td>0.81</td><td>0.79</td><td>0.83</td></tr>
    <tr><td>4</td><td>0.77</td><td>0.82</td><td>0.79</td><td>0.84</td></tr>
    <tr><td>5</td><td><strong>0.82</strong></td><td>0.79</td><td><strong>0.80</strong></td><td><strong>0.85</strong></td></tr>
  </tbody>
</table>

<hr/>

<h2>🧪 Sensitivity Analysis</h2>
<p>Grid over learning rate {<code>0.1</code>, <code>0.001</code>} × embedding dim {<code>8</code>, <code>128</code>} (5 epochs each):</p>
<table>
  <thead>
    <tr><th>LR</th><th>Emb Dim</th><th>Final ROC-AUC</th></tr>
  </thead>
  <tbody>
    <tr><td>0.1</td><td>8</td><td>0.81</td></tr>
    <tr><td>0.1</td><td>128</td><td>0.64</td></tr>
    <tr><td>0.001</td><td>8</td><td>0.69</td></tr>
    <tr><td>0.001</td><td>128</td><td><strong>0.82</strong></td></tr>
  </tbody>
</table>
<p>
Takeaways: overly high LR (0.1) with a large embedding can underperform; LR=1e-3 with 128-dim embeddings gives the most stable AUC.
</p>

<hr/>

<h2>📊 Reproducibility Notes</h2>
<ul>
  <li>Seeds fixed to <code>24</code> for <code>random</code>, <code>numpy</code>, <code>torch</code> and <code>PYTHONHASHSEED</code>.</li>
  <li>Strict dtypes from collate: <code>x, rev_x: Long</code>; <code>masks, rev_masks: Bool</code>; <code>y: Float</code>.</li>
  <li>Only true visits are reversed; padded rows remain at the tail with zero masks.</li>
  <li>Training loader shuffles; validation loader does not.</li>
</ul>

<hr/>

<h2>⚠️ Challenges &amp; How They Were Solved</h2>
<ol>
  <li><strong>Padding &amp; masking across two attentions:</strong> Constructed visit masks via <code>any</code> over codes and applied them in the final attention sum, preventing padded visits from contributing.</li>
  <li><strong>Reverse-time consistency:</strong> Reversed only the actual visits and mirrored masks to align the α/β attentions with true temporal order.</li>
  <li><strong>Vectorization (no loops):</strong> Used pure tensor ops for attention aggregation: <code>c = Σ (α ⊙ β ⊙ v ⊙ mask)</code>.</li>
  <li><strong>Stable training:</strong> Used probabilities (post-sigmoid) with <code>BCELoss</code>, and validated LR/embedding choices via sensitivity analysis.</li>
</ol>

<hr/>

<h2>🛠️ Environment</h2>
<pre><code>python &gt;= 3.9
pip install torch numpy scikit-learn
</code></pre>

<hr/>

<h2>🚀 How to Run</h2>
<ol>
  <li>Open <code>HW4_RETAIN.ipynb</code> and run cells top-to-bottom.</li>
  <li>Verify collate shapes/dtypes and DataLoader length (<code>25</code> for train).</li>
  <li>Train RETAIN for 5 epochs; confirm validation ROC-AUC &gt; <code>0.8</code> (target &gt; <code>0.7</code>).</li>
  <li>Run the sensitivity grid to compare LR/embedding settings.</li>
</ol>

<hr/>

<h2>📄 Reference</h2>
<p>
Choi et&nbsp;al., “RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism,” arXiv:1608.05745.
</p>
