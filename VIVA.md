# Comprehensive Viva Voce Guide: Stock Price Prediction Using Deep Learning
## MBA Data Science & Artificial Intelligence Academic Examination
### Project: *A Deep Learning-Based Time Series Analysis of ADANIPORTS Stock Prices (LSTM vs. GRU)*

---

### Q1: What is stock-price prediction?
**Answer:** Stock price prediction is the forecasting of future financial asset prices based on past historical price action, trading volumes, and technical trends. In quantitative finance, it is framed as a supervised time-series regression problem where models attempt to capture temporal dependencies and momentum patterns.

---

### Q2: Why did you choose ADANIPORTS for this study?
**Answer:** ADANIPORTS (Adani Ports and Special Economic Zone Ltd.) is India's largest private commercial port operator and a premier constituent of the NIFTY 50 index. Its historical price trajectory spans multiple macroeconomic regimes—including the 2008 global financial crisis, commodity booms, policy shifts, infrastructure growth, and the 2020 pandemic volatility—offering a rich, non-stationary dataset of over 3,300 trading days ideal for testing deep learning robustness.

---

### Q3: Why is stock price forecasting a time-series problem?
**Answer:** In traditional machine learning (e.g., predicting customer churn), observations are assumed to be Independent and Identically Distributed (i.i.d.). In stock markets, observations have strict temporal autocorrelation: today's price is deeply dependent on yesterday's price. The order of observations matters fundamentally; shuffling the sequence destroys temporal relationships.

---

### Q4: Why did you choose LSTM for this project?
**Answer:** Standard Recurrent Neural Networks (RNNs) suffer from the *vanishing gradient problem*, meaning they cannot retain memory over sequences longer than 5–10 steps. Long Short-Term Memory (LSTM) networks solve this by introducing an internal constant error carousel (the Cell State) and specialized gating mechanisms, enabling the network to learn both short-term fluctuations and long-term multi-month trend cycles.

---

### Q5: What is an LSTM and how does it work inside?
**Answer:** An LSTM is a specialized deep recurrent architecture governed by three mathematical gates:
1. **Forget Gate ($f_t$):** Uses a sigmoid function to decide what past information to throw away.
2. **Input Gate ($i_t$):** Decides which new information to store in the cell state.
3. **Output Gate ($o_t$):** Filters the internal cell state to emit the hidden state ($h_t$) passed to the next time step.

---

### Q6: What is a GRU (Gated Recurrent Unit)?
**Answer:** A GRU is a more modern, streamlined variation of the LSTM introduced by Cho et al. in 2014. It merges the cell state and hidden state into a single recurrent state and uses two gates:
1. **Reset Gate ($r_t$):** Controls how much previous state information to forget when computing new candidate memory.
2. **Update Gate ($z_t$):** Simultaneously determines how much old memory to retain and how much new candidate information to incorporate.

---

### Q7: What are the key differences between LSTM and GRU?
**Answer:**
- **Number of Gates:** LSTM has 3 gates (Forget, Input, Output); GRU has 2 gates (Reset, Update).
- **Internal States:** LSTM maintains separate Cell State ($c_t$) and Hidden State ($h_t$); GRU maintains only a Hidden State ($h_t$).
- **Parameter Count:** GRU has approximately 25% fewer trainable parameters than an LSTM with the same number of units ($3 \times d$ vs. $4 \times d$).
- **Speed & Generalization:** GRU trains faster and is often less prone to overfitting on smaller or noisy financial datasets.

---

### Q8: Why did you compare LSTM and GRU instead of using just one?
**Answer:** In managerial decision-making and data science practice, model benchmarking is crucial. Comparing LSTM and GRU under identical experimental conditions (identical chronological splits, sequence lengths, learning rates, and early stopping rules) allowed us to determine whether model complexity (LSTM's 3 gates) was necessary or whether a leaner architecture (GRU) offered superior empirical generalization on ADANIPORTS.

---

### Q9: What is a sequence or sliding window?
**Answer:** A sequence (or sliding window) is a contiguous slice of historical time steps fed as a single input tensor to the neural network. For example, a window of length 60 means the model receives prices from $[t-59, \dots, t]$ to predict the single target value at $t+1$. As time advances, the window slides forward by one trading session.

---

### Q10: Why did you use a 60-day window size?
**Answer:** 60 trading days corresponds to approximately 3 calendar months (one financial quarter). In market microstructure, quarterly earnings releases, corporate disclosures, and institutional rebalancing occur on a 3-month cycle. A 60-day window gives the recurrent gates sufficient temporal depth to identify quarterly momentum without being diluted by ancient, irrelevant noise.

---

### Q11: Why can we NOT randomly shuffle time-series data?
**Answer:** Randomly shuffling time-series data destroys chronological causality. If shuffled, the model would train on future prices (e.g., from 2020) to predict past prices (e.g., from 2015), creating catastrophic data leakage and artificially inflated, invalid performance. In finance, training must strictly precede testing in time.

---

### Q12: What is data leakage, and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as future test set values) inadvertently leaks into the training pipeline. We prevented data leakage by:
1. Splitting the dataset chronologically (first 80% train, remaining 20% test).
2. Fitting the `MinMaxScaler` **only on the training split**. The test split was only *transformed* using the pre-fitted parameters.
3. Calculating all technical indicators (moving averages, returns) using strictly backward-looking rolling windows.

---

### Q13: Why do we scale data before feeding it to deep learning models?
**Answer:** Neural networks use gradient descent. Unscaled stock prices (e.g., ₹200 vs. ₹1,200) cause large, erratic gradient updates, slow convergence, or exploding gradients. Furthermore, activation functions like tanh and sigmoid operate with maximum sensitivity between $-1$ and $+1$ or $0$ and $1$. Scaling standardizes the input scale, enabling stable, rapid gradient descent.

---

### Q14: What is MinMaxScaler and how does it work?
**Answer:** `MinMaxScaler` linearly transforms features into a bounded range, typically $[0, 1]$, using the formula:
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
where $X_{min}$ and $X_{max}$ are derived exclusively from the training dataset.

---

### Q15: What is Mean Absolute Error (MAE)?
**Answer:** MAE measures the average absolute difference between the predicted and actual stock prices:
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$
It is intuitive for business stakeholders because it is expressed directly in the original currency unit (Indian Rupees ₹).

---

### Q16: What is Root Mean Squared Error (RMSE)?
**Answer:** RMSE is the square root of the mean squared errors:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
Because errors are squared before averaging, RMSE penalizes large outlier forecast errors much more heavily than MAE. In trading, a single large forecast error can lead to a catastrophic margin call, making RMSE an essential risk metric.

---

### Q17: What is Mean Absolute Percentage Error (MAPE)?
**Answer:** MAPE expresses forecast error as an average percentage of actual prices:
$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
This allows comparing forecast accuracy across different price levels and different stocks regardless of nominal share price. In our test set, both models achieved an exceptional MAPE below 3.5%.

---

### Q18: What is the R² (Coefficient of Determination) Score?
**Answer:** $R^2$ indicates the proportion of variance in the actual stock prices that is predictable from the model:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
An $R^2$ of 1.0 represents a perfect fit; an $R^2$ of 0 indicates the model performs no better than predicting the historical average. Our models achieved $R^2 > 0.96$, showing strong tracking of directional trend movements.

---

### Q19: Which model performed better on the test set, and what were the exact numbers?
**Answer:** **GRU performed better than LSTM.**
- **GRU Test Metrics:**
  - **MAE:** ₹9.27
  - **RMSE:** ₹13.99
  - **MAPE:** 2.26%
  - **R² Score:** 0.9832
- **LSTM Test Metrics:**
  - **MAE:** ₹13.09
  - **RMSE:** ₹19.71
  - **MAPE:** 3.19%
  - **R² Score:** 0.9666

---

### Q20: Why did GRU perform better than LSTM in this experiment?
**Answer:** GRU achieved superior performance because:
1. **Lower Model Complexity:** Financial time-series contain substantial market noise. With fewer gates and parameters, GRU had a lower risk of memorizing noisy price fluctuations.
2. **Faster Memory Update:** The coupled reset and update gates allowed GRU to adapt more swiftly to abrupt price level shifts without being weighed down by a redundant separate cell state.

---

### Q21: What is overfitting in machine learning?
**Answer:** Overfitting occurs when a neural network learns the noise, idiosyncrasies, and random fluctuations of the training data rather than the true underlying distribution. An overfitted model achieves near-zero error on training data but fails miserably on unseen test data.

---

### Q22: What specific mechanisms did you implement to prevent overfitting?
**Answer:**
1. **Dropout Regularization:** Added 20% Dropout layers after each recurrent layer to randomly deactivate neurons during training.
2. **EarlyStopping:** Monitored validation loss with a patience of 10 epochs to halt training as soon as generalization error plateaued.
3. **Learning Rate Scheduling (`ReduceLROnPlateau`):** Reduced the learning rate by 50% when validation loss stagnated, avoiding overshooting minima.
4. **Restoring Best Weights:** Automatically restored the exact checkpoint weights that yielded the lowest validation loss.

---

### Q23: What is Dropout and how does it help recurrent networks?
**Answer:** Dropout is a regularization technique proposed by Srivastava et al. (2014) where a fraction (here, 20%) of neuron activations are randomly set to zero during each training step. This forces the remaining neurons to learn redundant, robust feature representations rather than co-adapting, which drastically curtails overfitting.

---

### Q24: What is EarlyStopping?
**Answer:** EarlyStopping is an automated training control callback. If the validation loss fails to improve for a preset number of epochs (the `patience` parameter), training terminates immediately and the best saved model weights are restored.

---

### Q25: Can this deep learning model guarantee the future stock price?
**Answer:** **Absolutely not.** In the finance literature, Eugene Fama's *Efficient Market Hypothesis* (EMH) states that market prices incorporate information rapidly. Stock prices are influenced by unforeseen macro events, corporate governance announcements, geopolitical tensions, regulatory changes, and institutional sentiment that cannot be captured solely from historical OHLC curves. Our model generates a quantitative statistical expectation, not a deterministic guarantee.

---

### Q26: What is VWAP and why is it important in stock analysis?
**Answer:** VWAP (Volume Weighted Average Price) is the ratio of the total value traded to the total volume traded over a session:
$$\text{VWAP} = \frac{\sum (\text{Price} \times \text{Volume})}{\sum \text{Volume}}$$
Institutional asset managers and algorithmic execution desks benchmark their large block trades against VWAP to measure whether they achieved favorable execution.

---

### Q27: How does your Flask web dashboard serve predictions without retraining?
**Answer:** We decoupled the **training pipeline (`train.py`)** from the **inference server (`app.py`)**. The training script pre-trains the neural networks and serializes the best Keras models (`lstm_model.keras`, `gru_model.keras`), fitted `scaler.pkl`, and evaluation metrics `model_results.json`. The Flask app loads these lightweight artifacts into memory, extracts the latest 60 days of quotes, runs a single forward pass in milliseconds, and inverse-transforms the forecast instantaneously.

---

### Q28: How would you extend this project in an enterprise or hedge fund setting?
**Answer:**
1. **Multimodal Data:** Integrate NLP sentiment analysis from financial news headlines (Bloomberg/Reuters) and social media (Twitter/X).
2. **Fundamental Metrics:** Incorporate quarterly P/E ratios, debt-to-equity, port cargo throughput volumes, and container tariff rates.
3. **Advanced Architectures:** Evaluate Bidirectional LSTMs, Temporal Convolutional Networks (TCN), or Transformer-based Time-Series Attention models (e.g., PatchTST).
4. **Reinforcement Learning:** Wrap predictions into a trading agent with slippage, transaction costs, and portfolio risk constraints (Sharpe ratio maximization).
