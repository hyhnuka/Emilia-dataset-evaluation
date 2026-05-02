# EMILIA: Evaluasi Dataset Q&A tentang Kesehatan Mental Indonesia

Proyek ini difokuskan pada evaluasi dataset tanya-jawab (Q&A) sintetis bertema psikologi. Evaluasi dilakukan menggunakan berbagai metrik termasuk *Natural Language Inference* (NLI) dan *LLM-as-a-Judge* untuk memastikan relevansi jawaban, meminimalkan halusinasi, dan menjaga kualitas dataset.

## Dataset Information

Dataset yang digunakan dalam proyek ini adalah dataset Q&A sintetis yang dirancang khusus untuk domain psikologi.

- **Dataset Name:** https://huggingface.co/datasets/hnuka/Emilia-Indonesian-Psychology-QnA-7K
- **Size:** ~7,500 pasangan Pertanyaan-Jawaban (Question-Answer pairs).
- **Sources:** Diolah dari 18 dokumen/buku referensi psikologi terpercaya.
- **Generation:** Menggunakan Large Language Models (GROK-3 & ChatGPT) dengan teknik *Prompt Engineering* yang dioptimalkan untuk menghasilkan dataset berupa QnA 
- **Language Support:** Indonesia (Utama), English, dan data hasil OCR.

## Project Structure

Berikut adalah struktur folder utama dalam proyek ini:

- `dataset/`: Berisi file dataset mentah dan gabungan (`combined_dataset_all.csv`) dalam berbagai kategori (idn, eng, idn-ocr).
- `NLI/`: Implementasi evaluasi menggunakan *Natural Language Inference* untuk memeriksa konsistensi logis antara dokumen sumber dan jawaban yang digenerate.
- `LLM-Judge/`:
    - `Answer Relevance/`: Skor evaluasi mengenai seberapa relevan jawaban terhadap pertanyaan yang diberikan.
    - `Hallucination/`: Skor evaluasi untuk mendeteksi adanya informasi yang tidak terdapat pada sumber (halusinasi model).
- `code/`: Berisi Jupyter Notebooks (`.ipynb`) yang digunakan untuk proses pengolahan data dan pipeline evaluasi.

## Evaluation Metrics

1. **Answer Relevance:** Mengukur apakah jawaban secara langsung menjawab inti pertanyaan tanpa informasi yang tidak perlu.
2. **Hallucination Detection:** Memastikan model tidak mengarang fakta yang tidak ada dalam dokumen referensi (Ground Truth).
3. **Natural Language Inference (NLI):** Mengkategorikan hubungan antara premis (sumber dokumen) dan hipotesis (pertanyaan dan jawaban) menjadi *entailment*, *neutral*, atau *contradiction*.


## Contributors

- **Haliza Nur Kamila Apalwan** (Informatics Engineering, ITS)
- Tim AI Research EMILIA

---
*Proyek ini merupakan bagian dari riset pengembangan AI Agent untuk dukungan kesehatan mental.*
