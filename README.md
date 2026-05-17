# Part 3: NLP and Sequence Modeling Mini Project

## Overview
This project builds a Natural Language Processing (NLP) pipeline to classify customer support messages into three sentiment categories: positive, neutral, and negative. I compared a traditional baseline model (TF-IDF with Logistic Regression) against a deep learning sequence model (LSTM).

## Project Structure
- `notebook.ipynb`: Contains text preprocessing, baseline modeling, and LSTM sequence modeling.
- `requirements.txt`: Python libraries required.
- `results/`: Contains the accuracy/loss plots (`model_evaluation.png`) and `sample_predictions.txt`.

## Task 3: Why Text Must Be Converted to Vectors
Machine learning models are essentially massive math equations. They cannot read English or understand words natively. Therefore, text must be converted into a numerical format (vectors) before being fed into a model. 
* Traditional methods like **TF-IDF** count the frequencies of words and weigh them by how unique they are across the dataset. 
* Modern approaches use **Word Embeddings** (which the LSTM in this project uses). Embeddings convert words into dense numerical vectors where words with similar meanings are placed closer together in space, allowing the model to actually learn the "context" of the word, not just its frequency.

## Task 6: Attention and Transformer Reflection
* **Why RNNs struggle with long-term dependencies:** Standard RNNs process text sequentially, one word at a time, updating a single hidden state. For long sentences, by the time the RNN reaches the end, the mathematical gradients start vanishing, causing the model to "forget" the context from the beginning of the sentence.
* **How LSTMs help with memory:** LSTMs fix this by introducing a "cell state" and three gates (Forget, Input, Output). These gates actively decide what information is useless and should be thrown away, and what information is important enough to carry forward down the line. 
* **What attention solves:** Even LSTMs bottleneck because they try to compress an entire sentence into one final vector. The Attention mechanism solves this by allowing the model to look at *every single word* in the input simultaneously and dynamically assign a "weight" to the words that matter most for predicting the next output.
* **Why transformers are important:** Transformers threw away the sequential recurrence (RNN/LSTM) entirely and rely solely on Attention mechanisms. Because they don't process words one-by-one, they are incredibly fast to train in parallel across GPUs. This massive scalability is what unlocked modern Generative AI models like ChatGPT and Gemini.
