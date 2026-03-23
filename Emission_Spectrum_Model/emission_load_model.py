import h5py 
from datetime import datetime
import copy
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from petitRADTRANS.planet import Planet
from sklearn.decomposition import PCA


plt.rcParams.update({
    'font.size': 22,          # Global font size
    'axes.labelsize': 24,     # X and Y label size
    'axes.titlesize': 24,     # Title size
    'xtick.labelsize': 22,    # X-axis tick numbers
    'ytick.labelsize': 22,    # Y-axis tick numbers
    'legend.fontsize': 18     # Legend text
})


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
startTime = datetime.now()


with h5py.File("emission_w_0.3_5.h5", "r") as f:  #calls data with RQMC sampling for training (2048 samples). Spans the sample space so better for training.
    wl_timeseries=f['emission_wl_0.3_5'][:]
wl_timeseries=wl_timeseries*1e4


with h5py.File("Emission_RQMC_combined.h5", "r") as f:
    X_temp = f['emission_input'][:]
    y_temp = f['emission_flux'][:]


with h5py.File("real_emission.h5","r") as f:  #calls data for WASP-121b for testing
    X_planet=f['real_emission_input'][:]
    y_planet=f['real_emission_flux'][:]


cols_to_log = [0,1,7]


X_planet [cols_to_log] = np.log(X_planet[cols_to_log])  #log transform the features that are log distributed for better training. Only temp, gravity and radius ratio are log distributed.
X_temp [:,cols_to_log] = np.log(X_temp[:, cols_to_log])

print(X_temp[0][0],X_temp[1][0])

print(np.shape(X_temp))

y_temp = y_temp.reshape(16384,3843) #need to train in log normal space

y_temp=y_temp[:,0:2814]

y_copy = y_temp.copy()

y_temp=np.log(y_temp)


y_planet = (y_planet.reshape(3843))  
y_planet=y_planet[0:2814] 

flux_shape=y_temp


shape_scaler = StandardScaler()
flux_shape = shape_scaler.fit_transform(flux_shape)

preprocessor=MinMaxScaler() 
X_temp=preprocessor.fit_transform(X_temp)   #normalises input features. RQMC isnt normally distributed by design, so can only really use MinMax Scaling. 


X_train, X_val, flux_shape_train, flux_shape_val = train_test_split(
    X_temp,  
    flux_shape,   
    test_size=0.2,
    random_state=42
)

################################################################




class LinearBlock(nn.Module):
    
    def __init__(self, in_features, out_features, dropout_rate=0.0, activation='leaky_relu'):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        # Batch Norm is typically applied before activation
        layers.append(nn.BatchNorm1d(out_features))

        if activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'leaky_relu':
            # Negative slope 0.01 is standard; helps 'dead' neurons recover
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        elif activation == 'relu':
            layers.append(nn.ReLU())
        
      
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, features, dropout_rate=0.0):  #tried dropout=0.0???
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.GELU(),   
            nn.LayerNorm(features),
            nn.Linear(features, features),    
            nn.GELU(),
            nn.LayerNorm(features),
            nn.Dropout(dropout_rate), 
        )
        

    def forward(self, x):
        return (x + self.block(x)) # The skip connection

class Shape_Model(nn.Module):
    def __init__(self):
        super().__init__() #try decrease the amount of noise          
        
       
        self.input_layer = LinearBlock(9, 512, dropout_rate=0.0, activation='gelu')
        self.hidden = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            #ResBlock(512),
            LinearBlock(512,1024, dropout_rate=0.0, activation='gelu'),   
        )    
        self.output_layer = nn.Linear(1024, 2814)


    def forward(self,x):
        x = self.input_layer(x)
        x = self.hidden(x)
        return self.output_layer(x) #standard feedforward architecture. No residual connections here as it wouldnt be helpful with learning. We are trying to approximate a complex shape. 
      
        #X_initial = self.input_layer(x)
        #X_hidden = self.hidden(X_initial)
        #return self.output_layer(X_initial + X_hidden)



# 1. Re-instantiate the model structure
loaded_model_shape = Shape_Model().to(device)

# 2. Load the state dictionary from the file
# Use map_location to ensure it loads correctly regardless of whether it was saved on GPU or CPU
state_dict_shape = torch.load('emis_shape_9.pth', map_location=device)

# 3. Load the weights into the model
loaded_model_shape.load_state_dict(state_dict_shape)

#4 load model in eval mode
loaded_model_shape.eval()

print("Models loaded successfully and set to eval mode.")

model_shape=loaded_model_shape



def predict_combined(model_shape, data):
    model_shape.eval()
    
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)  #fix data if not in correct format
    data = data.to(device)
    
    with torch.no_grad():
        # Get model predictions
        pred_shape_scaled = model_shape(data).cpu().numpy()
        pred_shape_scaled = shape_scaler.inverse_transform(pred_shape_scaled)
        pred_shape=pred_shape_scaled
   
        final_prediction = np.exp(pred_shape)

    return final_prediction, pred_shape  #final in norm units, else in log units
    #return final_prediction


with h5py.File("emission_test_better.h5", "r") as f:  #calls data with RQMC sampling for training (2048 samples). Spans the sample space so better for training.
    X_test=f['emission_input'][:]
    flux_test=f['emission_flux'][:]

flux_test=flux_test.reshape(1024,2814)


#log indices 0,1,7 already in dataset
X_test=preprocessor.transform(X_test)

preds_raw = { 'rescaled': [],'shape': []}
targets_raw = {'rescaled': [], 'shape': []}

r2= {'rescaled':[], 'shape': []}
rmse = {'rescaled':[], 'shape': []}

nrmse = {'rescaled':[], 'shape': []}

#print(np.shape(rad_compare_std))
for i in range(1024):
    pred=predict_combined(model_shape, X_test[i:i+1])

    preds_raw['rescaled'].append(pred[0][0].flatten())
    #preds_raw['shape'].append(pred[1][0]*pred[3][0]) #scaled shape at mean=0

    preds_raw['shape'].append(pred[1][0]) #compare shape in log units  
   

    targets_raw['rescaled'].append(flux_test[i,:].flatten())
    targets_raw['shape'].append(np.log(flux_test[i,:].flatten())) #compare shape in logspace? 
   
  
    r2['rescaled'].append(r2_score(targets_raw['rescaled'][-1].flatten(), preds_raw['rescaled'][-1].flatten()))
    r2['shape'].append(r2_score(targets_raw['shape'][-1].flatten(), preds_raw['shape'][-1].flatten()))

    rmse['rescaled'].append(root_mean_squared_error(targets_raw['rescaled'][-1].flatten(), preds_raw['rescaled'][-1].flatten()))
    rmse['shape'].append(root_mean_squared_error(targets_raw['shape'][-1].flatten(), preds_raw['shape'][-1].flatten()))
   
    nrmse['rescaled'].append(100*((root_mean_squared_error(targets_raw['rescaled'][-1].flatten(), preds_raw['rescaled'][-1].flatten()))/(np.mean(targets_raw['rescaled'][-1].flatten()))))
    nrmse['shape'].append((10**14)*(root_mean_squared_error(targets_raw['shape'][-1].flatten(), preds_raw['shape'][-1].flatten())/np.mean(targets_raw['rescaled'][-1].flatten())))

print('R2 - Rescaled:',np.min(r2['rescaled']), np.max(r2['rescaled']), np.mean(r2['rescaled']), np.median(r2['rescaled']))
print('R2 - Shape:',np.min(r2['shape']), np.max(r2['shape']), np.mean(r2['shape']), np.median(r2['shape']))

print('RMSE - Rescaled:',np.min(rmse['rescaled']), np.max(rmse['rescaled']), np.mean(rmse['rescaled']), np.median(rmse['rescaled']))
print('RMSE - Shape:', np.min(rmse['shape']), np.max(rmse['shape']), np.mean(rmse['shape']), np.median(rmse['shape']))

print('NRMSE - Rescaled:',np.min(nrmse['rescaled']), np.max(nrmse['rescaled']), np.mean(nrmse['rescaled']), np.median(nrmse['rescaled']))
print('NRMSE - Shape:', np.min(nrmse['shape']), np.max(nrmse['shape']), np.mean(nrmse['shape']), np.median(nrmse['shape']))

plt.hist(r2['shape'] , bins=100, label=r'$R^2$ Emission Model')
plt.title(r'$R^2$' ' Value Distribution in Log Space')
plt.ylabel('Frequency')
plt.xlabel(r'$R^2$')
plt.grid(visible=True)
plt.show()

plt.hist(r2['rescaled'] , bins=100, label=r'$R^2$ Emission Model')
plt.xlabel(r'$R^2$')
plt.ylabel('Frequency')
plt.grid(visible=True)
plt.title(r'$R^2$' ' Value Distribution in Linear Space')
plt.show()

plt.hist(nrmse['shape'] , bins=100, label='Normalised RMSE Log Space')
#plt.title('RMSE Value Distribution')
plt.ylabel('Frequency')
plt.xlabel(r'Difference [%] x$10^{-12}$')
plt.grid(visible=True)
plt.legend()
plt.show()  

plt.hist(nrmse['rescaled'] , bins=100, label='Normalised RMSE Linear Space')
#plt.title('RMSE Value Distribution')
plt.ylabel('Frequency')
plt.xlabel('Differenece [%]')
plt.grid(visible=True)
plt.legend()
plt.show()  


print(np.argmin(r2['shape']), np.argmin(r2['rescaled']))


plt.plot(wl_timeseries, (predict_combined(model_shape, X_temp[np.argmin(r2['rescaled']):np.argmin(r2['rescaled'])+1]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_copy[np.argmin(r2['rescaled'])], alpha = 0.6, color='b', label = 'Actual')
plt.show()
plt.plot(wl_timeseries, np.log(predict_combined(model_shape, X_temp[np.argmin(r2['rescaled']):np.argmin(r2['rescaled'])+1]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_temp[np.argmin(r2['rescaled'])], alpha = 0.6, color='b', label = 'Actual')
plt.show()


plt.plot(wl_timeseries, (predict_combined(model_shape, X_temp[np.argmin(r2['shape']):np.argmin(r2['shape'])+1]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_copy[np.argmin(r2['shape'])], alpha = 0.6, color='b', label = 'Actual')
plt.show()
plt.plot(wl_timeseries, np.log(predict_combined(model_shape, X_temp[np.argmin(r2['shape']):np.argmin(r2['shape'])+1]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_temp[np.argmin(r2['shape'])], alpha = 0.6, color='b', label = 'Actual')
plt.show()



planet = Planet.get('WASP-121 b')

#WASP 121-B parameter for testing
planet_real = preprocessor.transform(X_planet.reshape(1, -1))  #preprocess the real planet data with the same scaler used for training.


startTime2 = datetime.now()
WASP_predict=(predict_combined(model_shape, planet_real[0:1]))[0]
#WASP_predict=(predict_combined(model_shape, model_mean, model_std, planet_real[0:1]))[0]
endTime2=datetime.now()
print('Duration: {}'.format(endTime2 - startTime2)) 
   




plt.title('WASP-121 b Emission Spectrum',fontsize=24)
plt.plot(wl_timeseries, y_planet*1e-12, label='WASP-121 b', color='b')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Flux [TW ${cm}^{-2}$]')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.legend()
plt.show()


i=0
#plt.suptitle('Predicted vs. Actual Emission Spectra')
plt.subplot(2,2,1)
plt.title('Test Planet 1')
plt.plot(wl_timeseries, ((predict_combined(model_shape, X_test[i:i+1]))[0][0])*1e-12, color='r', label= 'Prediction')
plt.plot(wl_timeseries, (flux_test[i])*1e-12, alpha = 0.6, color='b', label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Flux [TW ${cm}^{-2}$]')
plt.legend()
i=7
plt.subplot(2,2,2)
plt.title('Test Planet 2')
plt.plot(wl_timeseries, ((predict_combined(model_shape, X_test[i+1:i+2]))[0][0])*1e-12, color='r', label= 'Prediction')
plt.plot(wl_timeseries, (flux_test[i+1])*1e-12, alpha = 0.6, color='b', label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Flux [TW ${cm}^{-2}$]')
plt.legend()

i=7
plt.subplot(2,2,3)
plt.title('Test Planet 3')
plt.plot(wl_timeseries, ((predict_combined(model_shape, X_test[i+2:i+3]))[0][0])*1e-12, color='r', label= 'Prediction')
plt.plot(wl_timeseries, (flux_test[i+2])*1e-12, alpha = 0.6, color='b', label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Flux [TW ${cm}^{-2}$]')
plt.legend()


plt.subplot(2,2,4)
plt.title('WASP-121 b')
plt.plot(wl_timeseries,(WASP_predict[0])*1e-12, color='r' ,label='Prediction')
plt.plot(wl_timeseries, (y_planet)*(1e-12), alpha=0.6,color='b',label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Flux [TW ${cm}^{-2}$]')
plt.legend()

plt.subplots_adjust(hspace=0.3)
plt.show()



plt.suptitle('Predicted vs. Actual Emission Spectra')
plt.subplot(2,2,1)
plt.title('Val Planet 1')
plt.plot(wl_timeseries, (predict_combined(model_shape, X_temp[0:1]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_copy[0], alpha = 0.6, color='b', label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux')
plt.legend()

plt.subplot(2,2,2)
plt.title('Test Planet 2')
plt.plot(wl_timeseries, (predict_combined(model_shape, X_temp[1:2]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_copy[1], alpha = 0.6, color='b', label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux')
plt.legend()

plt.subplot(2,2,3)
plt.title('Test Planet 3')
plt.plot(wl_timeseries, (predict_combined(model_shape, X_temp[2:3]))[0][0], color='r', label= 'Prediction')
plt.plot(wl_timeseries, y_copy[2], alpha = 0.6, color='b', label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux')
plt.legend()


plt.subplot(2,2,4)
plt.title('WASP-121 b')
plt.plot(wl_timeseries, (predict_combined(model_shape, X_temp[3:4]))[0][0], color='r', label='Prediction')
plt.plot(wl_timeseries, y_copy[3], alpha=0.6,color='b',label = 'Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux')
plt.legend()

plt.subplots_adjust(hspace=0.3)
plt.show()


