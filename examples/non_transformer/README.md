# Non-Transformer Model Examples

This folder contains examples of integrating non-transformer models with the MultiMind LLM interface. Each example demonstrates how to wrap and use a different type of model (statistical, classical ML, SSMs, RNNs, MLPs, diffusion, topological, etc.) using the `NonTransformerLLM` template and advanced wrappers.

## Included Example Types
- **Classical ML:** Scikit-learn (SVM, RF, GMM, DBSCAN, KMeans, CRF, HMM, etc.)
- **Deep Learning:** PyTorch (RNN, LSTM, GRU, Seq2Seq), Keras (CNN)
- **State Space Models:** Mamba, S4, Hyena, RWKV, Mega-S4, Liquid-S4, S4D, S4ND, DSS, GSS, MoE-Mamba, H3, RetNet, SE(3)-Hyena
- **Other Architectures:** MLP-only, Perceiver, Diffusion, Topological NNs
- **NLP Pipelines:** spaCy, NLTK, TextBlob, Gensim
- **AutoML:** CatBoost, LightGBM, XGBoost
- **Statistical Models:** ARIMA

## Example Table
| File                        | Model Type         | Description |
|-----------------------------|--------------------|-------------|
| advanced_usage.py           | Multi-model        | Register and use Mamba, Hyena, RWKV, CustomRNN with advanced features |
| mamba_hf_integration.py     | SSM (Mamba)        | Real HuggingFace Mamba integration |
| mamba_stub.py               | SSM (Mamba, stub)  | Stub for custom Mamba integration |
| hyena_stub.py               | SSM (Hyena, stub)  | Stub for custom Hyena integration |
| rwkv_stub.py                | SSM (RWKV, stub)   | Stub for custom RWKV integration |
| ssm_s4_mamba_stub.py        | SSM (S4/Mamba)     | Stub for S4/Mamba integration |
| s4d_stub.py, s4nd_stub.py   | SSM (S4D/S4ND)     | Stubs for S4D/S4ND variants |
| mega_s4_stub.py             | SSM (Mega-S4)      | Stub for Mega-S4 integration |
| liquid_s4_stub.py           | SSM (Liquid-S4)    | Stub for Liquid-S4 integration |
| dss_stub.py, gss_stub.py    | SSM (DSS/GSS)      | Stubs for DSS/GSS models |
| h3_stub.py                  | SSM (H3)           | Stub for H3 model |
| retnet_stub.py              | SSM (RetNet)       | Stub for RetNet model |
| se3_hyena_stub.py           | SSM (SE(3)-Hyena)  | Stub for equivariant Hyena |
| topological_nn_stub.py      | Topological NN     | Stub for topological deep learning |
| mlp_only_stub.py            | MLP-only           | Stub for MLP-Mixer/gMLP/HyperMixer |
| perceiver_stub.py           | Perceiver          | Stub for Perceiver/Perceiver IO |
| diffusion_text_stub.py      | Diffusion          | Stub for diffusion text models |
| moe_stub.py, moe_mamba_stub.py | MoE/MoE-Mamba   | Stubs for Mixture-of-Experts models |
| sklearn_example.py          | Classical ML       | Scikit-learn text classifier (MultinomialNB) |
| sklearn_svm_example.py      | Classical ML       | Scikit-learn SVM example |
| sklearn_rf_example.py       | Classical ML       | Scikit-learn Random Forest example |
| sklearn_gbm_example.py      | Classical ML       | Scikit-learn Gradient Boosting example |
| sklearn_logreg_example.py   | Classical ML       | Scikit-learn Logistic Regression example |
| sklearn_linreg_example.py   | Classical ML       | Scikit-learn Linear Regression example |
| sklearn_kmeans_example.py   | Classical ML       | Scikit-learn KMeans clustering |
| sklearn_dbscan_example.py   | Classical ML       | Scikit-learn DBSCAN clustering |
| sklearn_gmm_example.py      | Classical ML       | Scikit-learn Gaussian Mixture Model |
| sklearn_crf_example.py      | Classical ML       | Scikit-learn CRF example |
| hmmlearn_example.py         | Classical ML       | HMM example with hmmlearn |
| catboost_example.py         | AutoML             | CatBoost classifier example |
| lightgbm_example.py         | AutoML             | LightGBM classifier example |
| xgboost_example.py          | AutoML             | XGBoost classifier example |
| arima_example.py            | Statistical        | ARIMA time series model |
| pytorch_rnn_example.py      | Deep Learning      | PyTorch RNN text generator |
| pytorch_lstm_example.py     | Deep Learning      | PyTorch LSTM text generator |
| pytorch_gru_example.py      | Deep Learning      | PyTorch GRU text generator |
| pytorch_seq2seq_example.py  | Deep Learning      | PyTorch Seq2Seq model |
| keras_cnn_example.py        | Deep Learning      | Keras CNN text classifier |
| spacy_example.py            | NLP Pipeline       | spaCy pipeline integration |
| nltk_example.py             | NLP Pipeline       | NLTK pipeline integration |
| textblob_example.py         | NLP Pipeline       | TextBlob sentiment analysis |
| gensim_example.py           | NLP Pipeline       | Gensim topic modeling |

## Usage Instructions
- Each example is self-contained and runnable (see code comments for requirements).
- Most examples show how to wrap a model in a `NonTransformerLLM` or advanced subclass, and use it for generation or classification.
- See `advanced_usage.py` for how to register and use multiple models with advanced features (batching, streaming, adapters, chat memory, etc.).
- For SSMs and advanced models, see the stub files for extension points to plug in real research models.

## Extending These Examples
- To add your own model, subclass `NonTransformerLLM` or use one of the advanced templates (see `multimind/llm/non_transformer_llm.py`).
- Implement the required methods (`generate`, `chat`, etc.) for your model.
- Register your model with the MultiMindSDK registry and use it in chat, RAG, or agent workflows.
- For production, leverage advanced features: batch/streaming, adapters, config-driven instantiation, logging, and more.

---
For more details, see the MultiMindSDK documentation and the source code in `multimind/llm/non_transformer_llm.py`.
