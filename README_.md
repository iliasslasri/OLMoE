

To download some of the data:

```bash
wget -O data/wiki-001.json.gz "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/wiki/wiki-0001.json.gz?download=true"
```


- Scaling laws for tiny models: https://arxiv.org/pdf/2507.17702v1

OlMoE:

Ablations:
- Mixture-of-Experts vs. Dense
- Expert Granularity
- Shared Experts
- Load Balancing Loss
- Router Z-loss
- Loss free Routing
- Expert Choice vs. Token Choice X
- Sparse Upcycling X


MoE Analysis:
- Router Saturation
- Expert Co-activation
- Domain specialization
- Vocabulary specialization
