import pickle
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import models
import transformers

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DEFAULT_TEXT_INPUT = """This movie is really bad. Do not watch it!"""
CLASSES = ["negative", "positive"]

st.title("Sentiment Analysis")
st.markdown("**Stanford IMDB Large Movie Review Dataset**")

text_input = st.text_area(label="Input Text",
                                 value=DEFAULT_TEXT_INPUT,
                                 height=100)

st.sidebar.image("https://logos-download.com/wp-content/uploads/2016/10/Nvidia_logo.png", use_column_width=True)
st.sidebar.markdown("**Model selection**")
keras_models = ["MLP", "CNN", "LSTM"]
huggingface_models = ["distilroberta-base"]
model_name = st.sidebar.selectbox("Select model to use", keras_models+huggingface_models)

@st.cache(allow_output_mutation=True)
def load_tokenizer(model_name, tokenizer_path="./data/tokenizer.pickle"):
    if model_name in keras_models:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    elif model_name in huggingface_models:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer

@st.cache(allow_output_mutation=True)
def load_model_from_name(model_name="MLP"):
    model_choices = {
        "MLP": "./data/best_mlp.h5",
        "CNN": "./data/best_cnn.h5",
        "LSTM": "./data/best_lstm.h5",
        "distilroberta-base": "./data/tf_model.h5",
    }
    model_path = model_choices[model_name]
    if model_name in keras_models:
        # TODO: load keras model from path
        
        model = tf.keras.models.load_model(model_path)
    elif model_name in huggingface_models:
        # TODO: load huggingface model
        # hint: transformers.TFAutoModelForSequenceClassification.from_pretrained
        # you need the name of the original model, and your own weights
        # you can use model.load_weights for the latter
#         model.load_weights = model_path
#         config = transformers.AutoConfig.from_pretrained(model_name, output_hidden_states=True,
#                                             hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        model = transformers.TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return model

with st.spinner("Loading model"):
    tokenizer = load_tokenizer(model_name=model_name)
    model = load_model_from_name(model_name=model_name)
    tf.keras.utils.plot_model(model, to_file="./data/model.png", show_shapes=True, show_layer_names=True)
    image = Image.open("./data/model.png")
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    num_params = trainable_count + non_trainable_count
    num_params = str(int(num_params/1e6))+"M"
    caption = num_params + " param " + model_name
    st.sidebar.image(image, caption=caption, use_column_width=True)

def predict(model_name, model, input_text, max_seq_len=128):
    if model_name in keras_models:
        input_text = input_text.strip().lower()
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq,
                                                         padding="post",
                                                         maxlen=max_seq_len)
        preds = model.predict(input_seq)[0]
    elif model_name in huggingface_models:
        input_seq = [tokenizer.encode(input_text.strip())]
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq,
                                                         padding="post",
                                                         maxlen=max_seq_len)
        preds = model.predict(input_seq)[0][0]
    return preds

with st.spinner("Running inference"):
    start_time = time.time()
    # TODO: use the provided prediction function
    # what do you need to provide it? 
    logits = predict(model_name, model, DEFAULT_TEXT_INPUT, max_seq_len=128)
    end_time = time.time()

prediction = np.argmax(logits)
prediction = CLASSES[prediction]
st.markdown("### Prediction: "+prediction)
time_taken = str(round(end_time-start_time, 2))
st.markdown("Inference time: "+time_taken+" seconds")

with st.expander("See logits"):
    st.markdown(logits)
    fig, ax = plt.subplots()
    sns.barplot(x=CLASSES, y=logits)
    st.pyplot(fig)

with st.expander("Dataset citation"):
    st.markdown("""[**Stanford IMDB Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/)
    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [*Learning Word Vectors for Sentiment Analysis*](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).""")
