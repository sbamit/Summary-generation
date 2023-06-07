# Summary_generation
Application of Deep Learning models on the famous Cornell-newsroom dataset to extract summaries from news articles.
CIS5930 Project Overview
Authors: Sajib Biswas, Arunima Mandal, Fatema Tabassum Liza

The dataset is taken from the Cornell Newsroom dataset. The link to the dataset is given below.
https://lil.nlp.cornell.edu/newsroom/index.html 
You can go to the ‘download’ tab of the website and submit a form to get the dataset. Otherwise, we also shared a google drive link where the dataset will be hosted for a short period of time.

https://drive.google.com/file/d/1UeaNC4zP0gSEWtgPXdKnGcflsJz6o0xR/view?usp=sharing 

The program files are grouped in the following way.
Data preprocessing part is the most complicated and time-consuming part in this project. So, the program files created for this purpose are in a folder called ‘data_prep’ and are in .py format. These files need to run in the following order. The ordering of them is very important, because the output of one python program is used as the input for the next program. 

1. load_data_from_jsonl.py

2. filter_extractive_data_only.py

3. preprocessing_embed.py

4. preprocessing label_target.py

5. add_additional_features.py

6. add_domain_labels.py

7. train_test_split_embeddings_only.py

8. add_sent_num_to_train_test_split.py

Don’t forget to install the dependencies for this part, using the following commands. 

pip install sentence-transformers
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_lg


The machine learning model and result analysis are present in .ipynb formats. There are three notebook files for this purpose. One is (Modelling_withOUT_Sequential_Info.ipynb) for the machine learning models that run on sentence embeddings without sequential information, another is (Modelling_WITH_Sequential_Info.ipynb) for models which run on data with sequential information, and the third notebook (Analyze_Models_Results.ipynb) is for the result analysis and producing ROUGE scores. For these notebooks, we kept all the data in google drive and ran the notebooks on google colab free version. These should be pretty standard to run on google colab, while the data they work on must be uploaded to a google drive folder and configured accordingly. 

Thank you.




