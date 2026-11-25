---
title: Pronunciada
emoji: 🗣️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "6.32.0"
app_file: app.py
pinned: true
license: apache-2.0
short_description: On-device pronunciation assessment, ten languages
tags:
  - pronunciation
  - speech
  - language-learning
  - onnx
---

# Pronunciada demo

Scores a recording of a read sentence against its reference text and returns acceptance probabilities for each word and phoneme.

- Source and benchmarks: https://github.com/moglearn/pronunciada
- Install: `pip install pronunciada`

Recordings are scored on this Space and are not stored. The sample clips come from speechocean762,
OMPAL and FLEURS (CC BY 4.0). The acoustic model is ZIPA (CC BY 4.0).
