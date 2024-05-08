import tsfresh 
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tsfresh import extract_features
import timm
import librosa
import pywt

class config:
    MODEL = "tf_efficientnet_b0"
    LABEL_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']


NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

def spectrogram_from_eeg(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')

    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # # DENOISE
            # if USE_WAVELET:
            #     x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256,
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40
            img[:,:,k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0

    return img


USE_WAVELET = 'db8'
NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret

def features_from_eeg(eegs, display=False):
    for eeg in eegs:
        ids_data = np.array([], dtype=np.int64)
        times_data = np.array([])
        LLS = np.array([])
        LPS = np.array([])
        RRS = np.array([])
        RPS = np.array([])
        

        middle = (len(eeg)-10_000)//2
        eeg = eeg.iloc[middle:middle+10_000]

        signals_int = []
        LL = np.array([])
        LP = np.array([])
        RR = np.array([])
        RP = np.array([])
        for k in range(4):
            COLS = FEATS[k]
            for kk in range(4):

                # COMPUTE PAIR DIFFERENCES
                x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

                # FILL NANS
                m = np.nanmean(x)
                if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
                else: x[:] = 0

                # DENOISE
                if USE_WAVELET:
                    x = denoise(x, wavelet=USE_WAVELET)
                signals_int.append(x)

        signals = np.stack(signals_int, axis=0)
        LL = signals[0] + signals[1] + signals[2] + signals[3]
        LP = signals[4] + signals[5] + signals[6] + signals[7]
        RR = signals[8] + signals[9] + signals[10] + signals[11]
        RP = signals[12] + signals[13] + signals[14] + signals[15]

        id_data = np.full(10000, 1)
        time = np.arange(0, 10000, 1)

        LL = np.expand_dims(LL, axis=1)
        LP = np.expand_dims(LP, axis=1)
        RR = np.expand_dims(RR, axis=1)
        RP = np.expand_dims(RP, axis=1)

        ids_data = np.append(ids_data, id_data)
        times_data = np.append(times_data, time)
        LLS = np.append(LLS, LL)
        LPS = np.append(LPS, LP)
        RRS = np.append(RRS, RR)
        RPS = np.append(RPS, RP)

        signals = np.stack([ids_data, times_data, LLS, LPS, RRS, RPS], axis = 1)

        df = pd.DataFrame(signals)

        df.columns = ['id', 'time', 'LL', 'LP', 'RR', 'RP']
        df.set_index('time', inplace=True)
        df.reset_index(inplace=True)

        print('about to call tsfresh ')
        # Call TSFresh on the data
        settings = tsfresh.feature_extraction.settings.EfficientFCParameters()
        print('called settings')
        output = extract_features(df, column_id='id', column_sort='time', default_fc_parameters=settings)
        print('made it past tsfresh')
        return output


def get_rocket_output(rocket_model, parquet):
    pq = parquet
    middle = (len(pq)-2_000)//2
    pq = pq.iloc[middle:middle+2_000:2]
    #just doing this so the formatting works well
    pq = [pq.reset_index()]

    batch_size = 1
    x_batch = pd.concat(pq,keys=list(range(batch_size)),axis=0).reset_index(level=1)
    x_batch['instances'] = x_batch.index
    x_batch = x_batch.rename(columns={"level_1": "timepoints"})
    x_batch = x_batch.set_index(['instances', 'timepoints'])
    x_batch = x_batch.fillna(0)
    rocket_predictions = rocket_model.predict(x_batch)
    rocket_predictions = torch.from_numpy(rocket_predictions.to_numpy())
    return rocket_predictions  

def get_xgboost_output(xgboost_model,feature_row):
    x = feature_row.values.reshape(1, -1)
    xg_probs = xgboost_model.predict_proba(x)[0]
    return torch.tensor(xg_probs, dtype=torch.float32)

def EN_data_generation(parquet, spec_parquet):
    """
    Generates usable data from parquets
    """
    X = np.zeros((128, 256, 8), dtype='float32')
    y = np.zeros(6, dtype='float32')
    img = np.ones((128,256), dtype='float32')
    r = 0
    # spectrogram_file_path = self.paths.TRAIN_SPECTROGRAMS + str(row.spectrogram_id) + ".parquet"
    # spectrogram = pd.read_parquet(spectrogram_file_path).iloc[:,1:].values
    spectrogram = spec_parquet.iloc[:,1:].values
    # eeg_file_path = self.paths.TRAIN_EEGS + str(row.eeg_id) + ".parquet"
    eeg = parquet

    for region in range(4):
        img = spectrogram[r:r+300, region*100:(region+1)*100].T

        # Log transform spectrogram
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # Standarize per image
        ep = 1e-6
        mu = np.nanmean(img.flatten())
        std = np.nanstd(img.flatten())
        img = (img-mu)/(std+ep)
        img = np.nan_to_num(img, nan=0.0)
        X[14:-14, :, region] = img[:, 22:-22] / 2.0
        # img = eeg
        # X[:, :, 4:] = img

    return X


class EfficientNet(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(EfficientNet, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=False
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )

        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten())

        self.featurizer = True

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i:i+1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i:i+1] for i in range(4,8)]
        eegs = torch.cat(eegs, dim=1)

        # === Reshape (512,512,3) ===
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x,x,x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        if self.featurizer:
            x = self.adapter(x)
        else:
            x = self.custom_layers(x)
        return x
    
class FusionModel(nn.Module):
    def __init__(self, EN_Model, freeze = False):
        super(FusionModel, self).__init__()
        self.EfficientNet = EN_Model
        self.EN_out = 1280
        self.encoder_out = 48
        self.hidden = 10
        self.num_classes = 6
        if not freeze:
            for param in self.EfficientNet.parameters():
                param.requires_grad = True

        self.encoder = nn.Linear(self.EN_out, self.encoder_out)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_out + 6 + 6, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.num_classes)
        )

    def forward(self, X, rocket, xg, label=None):
        EN_features = self.EfficientNet(X)
        EN_features = F.relu(self.encoder(EN_features))
        try:
            combined_features = torch.cat((EN_features, rocket.squeeze(), xg), dim=1)
        except:
            # print(EN_features, rocket.squeeze(), xg)
            combined_features = torch.cat((EN_features, rocket.squeeze().unsqueeze(0), xg), dim=1)
            # raise
        out = self.classifier(combined_features)
        return F.softmax(out, dim = 1)