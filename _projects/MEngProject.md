---
layout: page
title: Improving Patient Safety Event Report Classification with Parameter Efficient Fine-tuned Language Model
description: MEng Project 2024-2025
img: /assets/img/MEngProject/thumbnail.png
importance: 2
category: University
---

University of Toronto

Focused on enhancing NLP model performance for patient safety applications, significantly
improving classification accuracy and reducing computational complexity.

<a href="https://www.mie.utoronto.ca/faculty_staff/eldan-cohen/" target="_blank" style="font-size: 20px; font-weight: bold;">Instructor: Prof. Eldan Cohen</a>

<!-- <a href="/assets/pdf/Thesis_Final_Report.pdf" target="_blank" style="font-size: 20px; font-weight: bold;">Thesis Report</a> -->

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MEngProject/thumbnail.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
</div>

```python 
r_values = [16,32,256]
lora_alphas = [32,64,512]
lora_dropouts = [0., 0.1]

for r in r_values:
    for alpha in lora_alphas:
        for dropout in lora_dropouts:
            trainer = run('roberta-large', learning_rate=1e-3, num_train_epochs=10, per_device_train_batch_size=32, r=r, lora_alpha=alpha,lora_dropout=dropout, type = 'lora')
```

