# Text Classification webapp using streamlit

![Machine learning expert and Data Scientist](https://github.com/shahzaibhamid/shahzaibhamid/blob/main/0_3HNWowLnPz9sqadH.jpg)

## This repository contains text classification with the help of four differnet models
1) MLP
2) CNN
3) LSTM
4) Hugging face model

To get started first you need to install the requirements from the requirement file,
You can build your docker file or just use pip install requirements.txt

After installing the requirements, open the train_keras file and run the complete code for one model, after that change the model and run the complete code again. This step will be repeated three time and at the end the best results of first three models will be saved.

The last model, HuggingFace uses the pretrained models and word vectors, therefore running it before is not necessary.
### Other than this all of the repository is complete, now go to the terminal in jupyter lab and run the streamlit app by using
streamlit run app.py
It will download the hugging face model, give it a little time for the first time.

## Later, you can make changes in this to develop your webapp and NLP techniques
