import gradio as gr
import warnings 
import joblib 
from sktime.utils import mlflow_sktime
from utils import *
from huggingface_hub import hf_hub_download
import shutil
import os

REPO_ID = "ramanodgers/HMSensemble"
FILENAME = "ensemble_9.pth"

xg_model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename="xgboost_model.pkl"))
weights = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="model")


description = ''' This is a Demo of our COMS 4995 Applied Computer Vision semester project: an ensemble method classification tool for EEG diagnoses.
Drag and drop an eeg and spectrogram parquet file and (hopefully) you'll get your result. Example files are available in the repo. 
The tsfresh feature extractor for the XGBoost model and our pytorch are not optimized for this small CPU environment and may take about a minute to run. 

Authors: Raman Odgers, Akhil Golla, Vinayak Kannan, Sohan Kshirsagar

'''
#this needs to be changed to load weights from HF and path the support models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rocket_model = mlflow_sktime.load_model(model_uri="./rocket/")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EN_model = EfficientNet(config)
Fusion_Model = FusionModel(EN_model, freeze=True).to(device)
Fusion_Model.to(device)

weights = torch.load(weights, map_location=device)
Fusion_Model.load_state_dict(weights)


def greet(eeg_file, spectrogram_file):

    parquet = pd.read_parquet(eeg_file.name) 
    spec = pd.read_parquet(spectrogram_file.name) 

    X = EN_data_generation(parquet, spec)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    rocket_in = get_rocket_output(rocket_model, parquet)

    feature_row = features_from_eeg([parquet], display=False)
    xg_out = get_xgboost_output(xg_model,feature_row).to(device)

    outputs = Fusion_Model(X, rocket_in, xg_out)
    _, preds = torch.max(outputs, 0)
    label = config.LABEL_COLS[preds.item()]

    return (label)


demo = gr.Interface(
    fn=greet,
    inputs=[gr.File(), gr.File()],

    outputs=[gr.Textbox(label="EEG class")],

    title="EEG Classification (APPLIED CV COMS 4995)",
    description = description, 
    live = False

)
demo.launch( share = True)