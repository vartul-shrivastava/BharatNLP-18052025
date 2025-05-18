# BharatNLP Multilingual Workbench

A comprehensive, browser-based NLP toolkit for working with Indian languages: Hindi, Marathi, Bengali, Tamil, and Telugu.

## 🚀 Features

- **Word Cloud Generator**: Visualize most frequent words in your text file (supports custom stopwords).
- **Topic Modeling**: Extract hidden topics using LDA, LSA, or NMF. Includes coherence tuning and best-topic-number finder.
- **Sentiment Analysis**: Label CSV data by sentiment; see class-wise word clouds and distributions.
- **Frequency Treemap**: Visualize token/n-gram frequencies as interactive treemaps (custom color scales supported).
- **N-gram Analysis**: Find and plot most common n-grams.
- **User-friendly Interface**: Drag-n-drop, smooth navigation, overlays, gradient scrollbars, and mobile support.
- **No Server-Side Image Compression**: Uploaded images retain full quality; visible border on images for clarity.

---

## ℹ️ About

**BharatNLP Multilingual Workbench** is developed for language researchers, students, and organizations working with Indian language corpora.  
Developed by Vartul Shrivastava and Prof. Dr. Shekhar Shukla.

---

## ❓ FAQs

### 1. Which languages are supported?
- Hindi, Marathi, Bengali, Tamil, Telugu (UI and NLP tasks).

### 2. What file formats are accepted?
- Plain text (.txt) and CSV (.csv). For CSV, you can pick the column to process.

### 3. Can I add custom stopwords?
- Yes! Upload a stopword file or paste your own list in the provided box.

### 5. Are my uploads private?
- Yes. All processing happens in your browser or your own server; no data is sent elsewhere.

### 6. Why isn’t my word cloud showing all words?
- Try adjusting the “max words” parameter or check your stopword settings.

### 7. How do I find the best number of topics in topic modeling?
- Use the “Find Best #Topics” button under Topic Modeling > Coherence Tuning.

---

## 🛠️ Installation & Setup

### Requirements

- Python 3.8+
- Flask
- Gensim, Numpy, Pandas, Matplotlib, scikit-learn, wordcloud, and other NLP libraries
- (Optional) Node.js if extending JS front-end

### Quick Start

1. **Clone the repo**  
   `git clone https://github.com/your-username/bharatnlp-workbench.git`
2. **Install dependencies**  
   `pip install -r requirements.txt`
3. **Run the app**  
   `python app.py`
4. **Open in your browser**  
   Go to [http://localhost:5000](http://localhost:5000)

---

## 🐞 Debugging Guide

- **Front-end JS issues**:  
  Use your browser console (F12) to check errors.  
  Make sure all scripts load (especially PapaParse, Plotly, Chart.js).

- **Back-end (Flask) issues**:  
  Watch terminal for error logs.  
  Typical problems:
  - Missing dependencies → run `pip install -r requirements.txt`
  - Permission errors → check file upload folder permissions

- **File upload issues**:  
  Confirm file format is supported (.txt or .csv).
  For CSV, ensure you select the column name after uploading.

- **Topic modeling errors**:  
  Large files may cause memory errors. Try smaller data or increase RAM.
  If coherence plot doesn't appear, check backend logs for missing data.

---

## 🙏 Acknowledgements

Thanks to open-source contributors and the NLP community!

---

## 📬 Contact

For feedback, feature requests, or support, open an [issue](https://github.com/your-username/bharatnlp-workbench/issues) or email the authors.

---

© 2024 BharatNLP Workbench
