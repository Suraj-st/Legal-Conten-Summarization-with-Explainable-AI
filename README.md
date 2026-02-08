# Legal-Conten-Summarization-with-Explainable-AI
Legal Conten Summarization and conver to with Explainable AI

## How to Run the Transfer Learning Application

1. Clone the Repository
2. Go inside the "Transfer learning app" folder
3. Activate the Python virtual environment
4. run the requirements.txt file
5. First Run the data_ext_src.py file. It will create a proper dataset to train the model.
6. The data_ext_src.py script will create legal_dataset_hf.json file
7. Then run the finetuned-base-model-train.py file to train the model using the prepared data.
8. The finetuned-base-model-train.py will generate legal-summarizer-final model with best checkpoint.
9. Then run the xai_app_upd.py file to start the application. (streamlit run xai_app_upd.py)

## How to Run the GENAI Application
1. Clone the Repository
2. Go inside the "Genai app" folder
3. Update your OPEN AI API key inside the .env file
4. Activate Python environment
5. Run the requirements.txt file
6. Run the leg_sum_xai_lst_app.py file (streamlit run leg_sum_xai_lst_app.py)



