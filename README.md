# BharatNLP: A Multilingual Text Visualization and Assessment NLP Toolkit for Prominent Indian Languages

A robust, browser-based toolkit for natural language processing (NLP) and visualization in Indian languages: **Hindi, Marathi, Bengali, Tamil, and Telugu**.

---

## Key Features

- **Word Cloud Generator:** Create word clouds that highlight the most frequent words in your dataset. Supports language-specific and custom stopword lists.
- **Topic Modeling:** Extract thematic structure using Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), or Non-Negative Matrix Factorization (NMF). Includes coherence score visualization and automated optimal topic selection.
- **Sentiment Analysis:** Classify textual data by sentiment (positive, neutral, negative) using machine translation and widely adopted English sentiment tools (TextBlob, VADER). View sentiment distributions and class-wise word clouds.
- **Frequency Treemap:** Visualize word or n-gram frequencies as hierarchical, interactive treemaps. Custom color maps and interactivity supported.
- **N-gram Analysis:** Identify and plot the most common n-grams (sequences of n words) within your corpus, with customization for n-gram size and output count.
- **User Interface:** Supports drag-and-drop file upload, column selection for CSV, smooth navigation, responsive layout, and touch/mobile compatibility.
- **Data Privacy:** All text processing is performed locally or on your designated server. No uploaded data is shared or transmitted externally.
- **High-Quality Visual Output:** Generated images retain full quality; clear image borders for presentation purposes.

---

## About

**BharatNLP Multilingual Workbench** is designed for researchers, educators, and organizations engaged in text analytics of Indian language corpora. The toolkit enables non-programmers and technical users alike to perform comprehensive NLP tasks and visualizations, helping bridge the gap between linguistic diversity and digital analysis tools.

- Developed by: [Vartul Shrivastava](mailto:vartul.shrivastava@gmail.com) & [Prof. Dr. Shekhar Shukla](mailto:shekhars@iimidr.ac.in)
- Repository: [GitHub Link](https://github.com/vartul-shrivastava/BharatNLP-18052025)

---

## Frequently Asked Questions (FAQ)

### 1. **Which languages are currently supported?**
BharatNLP supports Hindi, Marathi, Bengali, Tamil, and Telugu for both the user interface and NLP operations.

### 2. **What types of files can I analyze?**
You can upload plain text files (`.txt`) or comma-separated value files (`.csv`). For CSVs, you can select which column to analyze after uploading.

### 3. **Can I define my own stopwords?**
Yes. You may upload a file containing custom stopwords or paste your stopword list directly into the input box on the interface.

### 4. **Does the toolkit work offline or require an internet connection?**
The core analysis and visualization can run entirely in your browser or on your local server (depending on setup). Sentiment analysis using machine translation may require an internet connection for translation APIs.

### 5. **Is my data private and secure?**
Yes. All processing is performed locally (in-browser or on your server). No uploaded data leaves your environment.

### 6. **Why are some words missing from my word cloud or n-gram chart?**
This could be due to active stopword filtering, the "max words" parameter, or minimum frequency thresholds. Check your settings and adjust as needed.

### 7. **How can I determine the optimal number of topics for topic modeling?**
Use the “Find Best #Topics” option under the Topic Modeling section. The toolkit automatically calculates coherence scores and suggests the best topic count.

### 8. **Can I change the visualization appearance?**
Yes. Options are provided for changing fonts, color schemes, background, and word count for word clouds and treemaps.

### 9. **Do I need programming experience to use BharatNLP?**
No. The toolkit is designed for ease of use, with a graphical user interface requiring no coding.

### 10. **Are there limitations on file size or length?**
While there is no hardcoded limit, extremely large datasets may cause memory issues depending on your hardware. For best performance, start with moderate-sized files.

### 11. **What should I do if my CSV file is not processed correctly?**
Ensure your CSV is UTF-8 encoded and contains the target text in a dedicated column. After uploading, use the column picker to select the text column.

### 12. **Can I extend or contribute to the project?**
Absolutely! The code is open-source and contributions are welcome via pull requests. Please review the [CONTRIBUTING.md](./CONTRIBUTING.md) if available.

---

## Installation & Setup

### Requirements

- Python 3.8 or above
- Flask web framework
- Python libraries: `gensim`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `wordcloud`, `deep-translator`, `textblob`, `vaderSentiment`
- (Optional for development) Node.js and npm for front-end dependencies

### Quick Start

1. **Clone the repo**  
   `git clone https://github.com/vartul-shrivastava/BharatNLP-18052025`
2. **Install dependencies**  
   `pip install -r requirements.txt`
3. **Run the app**  
   `python app.py`
4. **Open in your browser**  
   Go to the localhost URL shown in terminal.

---

### **Debugging and Troubleshooting**

```markdown
## Debugging and Troubleshooting

**Front-end JavaScript issues:**
- Use your browser’s developer console (F12) to identify errors.
- Ensure required libraries (PapaParse, Plotly, Chart.js) are loaded.

**Back-end (Flask) issues:**
- Monitor the terminal for error logs.
- Common issues include missing dependencies (`pip install -r requirements.txt`) and permission errors in the upload folder.

**File upload issues:**
- Verify the file format is `.txt` or `.csv`.
- For CSV files, confirm that you select the correct text column after uploading.

**Large file or memory errors:**
- Topic modeling and some visualizations may require significant RAM for large files.
- Try using a smaller dataset or increase available system memory.

**Visualization not appearing:**
- Double-check input parameters, stopword lists, and browser compatibility.
- If issues persist, inspect backend logs for error details.

## Acknowledgements

This project makes use of several open-source libraries and datasets from the global NLP community. Special thanks to all contributors and resource maintainers whose work supports multilingual language processing.

## Contact & Support

For questions, feature requests, or to report issues, please contact the authors directly at:

- vartul.shrivastava@gmail.com
- shekhars@iimidr.ac.in

---

BharatNLP Multilingual Workbench | MIT Licensed
