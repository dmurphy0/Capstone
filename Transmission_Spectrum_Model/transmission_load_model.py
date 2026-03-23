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
from scipy import stats


plt.rcParams.update({
    'font.size': 22,          # Global font size
    'axes.labelsize': 24,     # X and Y label size
    'axes.titlesize': 24,     # Title size
    'xtick.labelsize': 22,    # X-axis tick numbers
    'ytick.labelsize': 22,    # Y-axis tick numbers
    'legend.fontsize': 18     # Legend text
})


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  #uses GPU if available
startTime = datetime.now()

R_jup = (const.R_jup.value)*100 #covert from mks to cgs units


with h5py.File('wl_trans.h5', 'r') as f:  
    wl_timeseries=f['wl_trans_0.3_15'][:]  #loads wavelength spacing for plots
wl_timeseries=wl_timeseries[0]*1e4 #rescales to microns

with h5py.File("data_1d_train.h5", "r") as f: #calls sample data with normal distribution for testing (800 samples)
    X_train_scaler= f['train_1D_x_train_data'][:]  #input parameters
    y_train_scaler= f['train_1D_y_train_data'][:]  #simulated transmission spectra

with h5py.File("data_1d_test.h5", "r") as f:  #calls sample data with normal distribution for testing (200 samples)
    X_test= f['train_1D_x_train_data'][:]
    y_test= f['train_1D_y_train_data'][:]

#legacy issue where both testing sets aren't combined. Next lines recombines these
X_compare = np.concatenate([X_train_scaler, X_test], axis=0)
rad_compare= (np.concatenate([y_train_scaler, y_test], axis=1)[0,:,0,:])/R_jup  #model unseen data for scoring the accuracy of model. 
rad_compare_mean, rad_compare_std = rad_compare.mean(axis=1, keepdims=True), rad_compare.std(axis=1, keepdims=True)


rad_test = y_test[0, :, 0, :]/R_jup #reshapes radius data and scales to R_jupiter units
rad_test_mean,rad_test_std= rad_test.mean(axis=1,keepdims=True), rad_test.std(axis=1,keepdims=True)
rad_test=(rad_test-rad_test_mean)/rad_test_std

rad_test_stats=np.concatenate([rad_test_mean,rad_test_std],axis=1) #used to renormalise data for comparison with predictions.

with h5py.File("RQMC_transmission.h5", "r") as f:  #calls data with RQMC sampling for training (2048 samples). Better span of parameter space so better for training.
    X_train=f['RQMC_transmission_input'][:]
    y_train=f['RQMC_transmission_radii'][:] 

with h5py.File("data_1d_realplanet.h5","r") as f:  #calls data for WASP-121b for testing
    y_planet=f['train_1D_y_train_data'][:]

real_radius= y_planet[0,:,0,:]
rad_train=(y_train[:,0,:]).reshape(2048,3912)
rad_combined=rad_train/R_jup    #reshape data and scale to jupiter radii units. 

#scale radii for predictions
rad_mean,rad_std=rad_combined.mean(axis=1,keepdims=True), rad_combined.std(axis=1,keepdims=True)
rad_combined=(rad_combined - rad_mean)/rad_std 

#scale mean and std for prediciton
X_train_copy=X_train.copy()
rad_scalar_X=(X_train_copy[:,1:2]/R_jup)
rad_mean= rad_mean - rad_scalar_X   #subtract planet radius so model can learn mean easier.

rad_mean_min, rad_mean_max = np.min(rad_mean), np.max(rad_mean)

#shift mean to 0 to avoid negative values after scaling
rad_mean = (rad_mean - rad_mean_min)/(rad_mean_max-rad_mean_min) #min-max scale mean to be between 0 & 1.


rad_std=rad_std*(X_train[:,2:3]/X_train[:,0:1]) #scale std by inverse scale height for easier learning. This greatly shifts the distribution of std values.
#standard deviation is a function of scale heigt, and so is a function of the ratio of 2 of our input parameters. Models find these types of relationships difficult to learn, so we must help it a little bit.


pt = PowerTransformer(method='yeo-johnson') #this pushed standard deviation dataset to be a normal distribution, this pushes prediction accuracy slightly further.  
rad_std= pt.fit_transform(rad_std.reshape(-1, 1))  #transform standard deviation to be normally distributed for easier learning.


X_combined=X_train
X_train, X_val, rad_shape_train, rad_shape_val, rad_mean_train, rad_mean_val, rad_std_train, rad_std_val = train_test_split(
    X_combined, 
    rad_combined,   
    rad_mean,
    rad_std,      
    test_size=0.1,
    random_state=42  #random state for replicability 
) #splits data to training and validation sets. 


preprocessor=MinMaxScaler()
X_train=preprocessor.fit_transform(X_train)   #normalises input features. RQMC isnt normally distributed by design, so can only really use MinMax Scaling. 
X_val=preprocessor.transform(X_val)

X_test_rad=X_test.copy()[:,1:2]/R_jup 

X_test = preprocessor.transform(X_test)
X_compare = preprocessor.transform(X_compare) #test set

#scales all data-set input parameters using minmax scaler in accordance with the min and max values of the X_train data set.

################################################################

class LinearBlock(nn.Module):
    
    def __init__(self, in_features, out_features, dropout_rate, activation):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        # Batch Norm is typically applied before activation
        layers.append(nn.BatchNorm1d(out_features))
        if activation == 'gelu':
            #layers.append(nn.GELU())
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(negative_slope=0.01))
      
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class Shape_Model(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.input_layer = LinearBlock(5, 512, dropout_rate=0.0, activation='gelu')
        self.hidden = nn.Sequential(
            LinearBlock(512, 1024, dropout_rate=0.0, activation='gelu'),
            LinearBlock(1024,2048, dropout_rate=0.05, activation='gelu'),
            LinearBlock(2048, 2048, dropout_rate=0.1, activation='gelu'),    
            LinearBlock(2048, 4096, dropout_rate=0.1, activation='gelu')
        )
        self.output_layer = nn.Linear(4096, 3912)
        

    def forward(self,x):
        x = self.input_layer(x)
        x = self.hidden(x)
        return self.output_layer(x) #standard feedforward architecture. No residual connections here as it wouldnt be helpful with learning. We are trying to approximate a compex shape. 
      

class Mean_Model(nn.Module): #this predicts the mean
    def __init__(self):
        super().__init__()
        self.input_layer = LinearBlock(5, 128, dropout_rate=0.0, activation='gelu')
        self.hidden = nn.Sequential(
            LinearBlock(128, 256, dropout_rate=0.0, activation='gelu'),
            LinearBlock(256,512, dropout_rate=0.05, activation='gelu'),
            LinearBlock(512,256,dropout_rate=0.05, activation='gelu'),
            LinearBlock(256,128,dropout_rate=0.0, activation='gelu')
        )
        self.output_layer = nn.Sequential(
            LinearBlock(128,64,dropout_rate=0.0, activation='gelu'),
            nn.Linear(64, 1)
        )

    def forward(self,x):
        X_initial = self.input_layer(x)
        X_hidden = self.hidden(X_initial)
        return self.output_layer(X_initial + X_hidden)  #residual connection to help with learning. Also helps to prevent vanishing gradients in deeper networks.
    

class Std_Model(nn.Module): #this predicts the mean
    def __init__(self):
        super().__init__()
        self.input_layer = LinearBlock(5, 128, dropout_rate=0.0, activation='gelu') 
        self.hidden = nn.Sequential(
            LinearBlock(128, 256, dropout_rate=0.0, activation='gelu'),
            LinearBlock(256,512, dropout_rate=0.1, activation='gelu'),
            LinearBlock(512,512, dropout_rate=0.1, activation='gelu'),
            LinearBlock(512,256, dropout_rate=0.1, activation='gelu'),
            LinearBlock(256,128,dropout_rate=0.0, activation='gelu')
        )
        self.output_layer = nn.Sequential(
            LinearBlock(128,64,dropout_rate=0.0, activation='gelu'),
            nn.Linear(64, 1)
        )

 
    def forward(self,x):
        X_initial = self.input_layer(x)
        X_hidden = self.hidden(X_initial)   
        return self.output_layer(X_initial + X_hidden)   #residual network connection to help with learning. Also helps to prevent vanishing gradients in deeper networks.
     

val_loss_plot = []
train_loss_plot=[]


# 1. Re-instantiate the model structure
loaded_model_shape = Shape_Model().to(device)
loaded_model_mean = Mean_Model().to(device)
loaded_model_std = Std_Model().to(device)

# 2. Load the state dictionary from the file
# Use map_location to ensure it loads correctly regardless of whether it was saved on GPU or CPU
state_dict_shape = torch.load('transmission_shape.pth', map_location=device)
state_dict_mean = torch.load('transmission_mean.pth', map_location=device)
state_dict_std = torch.load('transmission_std.pth', map_location=device)

# 3. Load the weights into the model
loaded_model_shape.load_state_dict(state_dict_shape)
loaded_model_mean.load_state_dict(state_dict_mean)
loaded_model_std.load_state_dict(state_dict_std)

# 4. CRITICAL: Set to evaluation mode for inference
loaded_model_shape.eval()
loaded_model_mean.eval()
loaded_model_std.eval()

print("Models loaded successfully and set to eval mode.")

model_shape=loaded_model_shape
model_mean=loaded_model_mean
model_std=loaded_model_std



def predict_combined(model_shape, model_mean, model_std, data):
    model_shape.eval()
    model_mean.eval()
    model_std.eval()
    
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)  #fix data if not in correct format
    data = data.to(device)
    
    with torch.no_grad():
        pred_shape = model_shape(data).cpu().numpy()
        pred_mean_scaled = model_mean(data).cpu().numpy()
        pred_std_scaled = model_std(data).cpu().numpy()   #calls models to predict

        data_copy=preprocessor.inverse_transform(data)   #inverse transform of original features for rescaling mean and std values. This is just the analytical inverse as these are seen values.
       
        pred_mean = pred_mean_scaled * (rad_mean_max - rad_mean_min) + rad_mean_min + data_copy[0,1]/R_jup # Inverse of min-max scaling, and re-shift by planet radius.

        pred_std = pt.inverse_transform(pred_std_scaled)* data_copy[0,0]/data_copy[0,2] #inverse power-transform for std, and then rescale by scale height. 
       
        final_prediction = (pred_shape * pred_std) + pred_mean
        
    return final_prediction, pred_shape, pred_mean, pred_std


preds_raw = { 'rescaled': [],'shape': [], 'mean': [], 'std': []}
targets_raw = {'rescaled': [], 'shape': [], 'mean': [], 'std': []}
r2= {'rescaled':[], 'shape': []}
rmse = {'rescaled':[], 'shape': [], 'mean': [], 'std': []}
nrmse = {'rescaled':[], 'shape': []}

for i in range(1000):
    pred=predict_combined(model_shape, model_mean, model_std, X_compare[i:i+1])

    preds_raw['rescaled'].append(pred[0][0].flatten())
    preds_raw['shape'].append(pred[1][0]*pred[3][0]) #scaled shape at mean=0
    #preds_raw['shape'].append(pred[1][0]*rad_compare_std[i:i+1]+rad_compare_mean[i:i+1]) #shape is still normalised, unnormalise with analytical values for mean nd std for comparison in jupiter radii units and not std units. 
    preds_raw['mean'].append(pred[2][0])   
    preds_raw['std'].append(pred[3][0])  #these are all unnormalised

    targets_raw['rescaled'].append(rad_compare[i:i+1].flatten())
    #targets_raw['shape'].append(rad_compare[i:i+1].flatten())  #same mean and std as rescaled, so comparison is purely on shape learning.
    targets_raw['shape'].append(rad_compare[i:i+1].flatten()-rad_compare_mean[i:i+1]) #specturm moved to mean=0 for comparison of shape*std
    targets_raw['mean'].append(rad_compare_mean[i:i+1])
    targets_raw['std'].append(rad_compare_std[i:i+1]) 
   
    r2['rescaled'].append(r2_score(targets_raw['rescaled'][-1].flatten(), preds_raw['rescaled'][-1].flatten()))
    r2['shape'].append(r2_score(targets_raw['shape'][-1].flatten(), preds_raw['shape'][-1].flatten()))

    rmse['rescaled'].append(root_mean_squared_error(targets_raw['rescaled'][-1].flatten(), preds_raw['rescaled'][-1].flatten()))
    rmse['shape'].append(root_mean_squared_error(targets_raw['shape'][-1].flatten(), preds_raw['shape'][-1].flatten()))
    rmse['mean'].append(root_mean_squared_error(targets_raw['mean'][-1].flatten(), preds_raw['mean'][-1].flatten()))
    rmse['std'].append(root_mean_squared_error(targets_raw['std'][-1].flatten(), preds_raw['std'][-1].flatten()))

    nrmse['rescaled'].append(100*((root_mean_squared_error(targets_raw['rescaled'][-1].flatten(), preds_raw['rescaled'][-1].flatten()))/(np.mean(targets_raw['rescaled'][-1].flatten()))))
    nrmse['shape'].append(100*(root_mean_squared_error(targets_raw['shape'][-1].flatten(), preds_raw['shape'][-1].flatten())/np.mean(targets_raw['rescaled'][-1].flatten())))


print('R2 - Rescaled:',np.min(r2['rescaled']), np.max(r2['rescaled']), np.mean(r2['rescaled']))
print('R2 - Shape:',np.min(r2['shape']), np.max(r2['shape']), np.mean(r2['shape']))
print('RMSE - Rescaled:',np.min(rmse['rescaled']), np.max(rmse['rescaled']), np.mean(rmse['rescaled']))
print('RMSE - Shape:', np.min(rmse['shape']), np.max(rmse['shape']), np.mean(rmse['shape']))
print('RMSE - Mean:',np.min(rmse['mean']), np.max(rmse['mean']), np.mean(rmse['mean']))
print('RMSE - Std:',np.min(rmse['std']), np.max(rmse['std']), np.mean(rmse['std']))
print('Nrmse - Rescaled:', np.min(nrmse['rescaled']), np.max(nrmse['rescaled']), np.mean(nrmse['rescaled']),np.median(nrmse['rescaled']))


print(np.argmin(r2['rescaled']))

plt.hist(r2['shape'] , bins=100, label=r'$R^2$ Transmisison Model')
plt.title(r'$R^2$' ' Value Distribution')
plt.ylabel('Frequency')
plt.xlabel(r'$R^2$')
plt.grid(visible=True)
plt.legend()
plt.show()

plt.hist(rmse['shape'] , bins=100, label='RMSE')
plt.title('RMSE Value Distribution')
plt.ylabel('Frequency')
plt.xlabel(r'[$\rm R_{Jup}$]' )
plt.grid(visible=True)
plt.legend()
plt.show()  

plt.hist(nrmse['rescaled'] , bins=100, label='Normalised RMSE')
plt.title('RMSE Value Distribution')
plt.ylabel('Frequency')
plt.xlabel('Differenece [%]')
plt.grid(visible=True)
plt.legend()
plt.show()  


planet=Planet.get('WASP-121 b')
planet_temp=planet.equilibrium_temperature
planet_radius = planet.radius
planet_reference_gravity =planet.reference_gravity
planet_metallicity = 0.5
planet_co = 0.3
planet_real=[planet_temp, planet_radius, planet_reference_gravity, planet_metallicity, planet_co]
planet_real=preprocessor.transform([np.array(planet_real)])


real_radius_initial = real_radius.copy()/R_jup
real_radius= (real_radius - np.mean(real_radius[0]))/(np.std(real_radius[0]))


startTime2 = datetime.now()
WASP_predict=(predict_combined(model_shape, model_mean, model_std, planet_real[0:1]))[0]
print('Duration: {}'.format(datetime.now() - startTime2))   #time length for 1 prediction. Approx 0.005seconds << 5seconds 
#petitRADTANS has no GPU support, likely much greater acceleration from GPU in just computing the prediction from the already trained model. 


print('WASP(Real,Pred):', np.mean(real_radius_initial[0]), np.std(real_radius_initial[0]), np.std(WASP_predict[0]), np.mean(WASP_predict[0]))


i=0  #use i=0,3,6,9... for different test planets. 

plt.figure(figsize=(24, 24))
plt.suptitle('Model Predcition vs. Actual Spectrum', fontsize = 16)

plt.subplot(2,2,1)
plt.title(f'Test Planet {i+1}')
pred=predict_combined(model_shape, model_mean, model_std, X_test[i:i+1])[0]
plt.plot(wl_timeseries,pred[0], color='r' ,label='Prediction')
plt.plot(wl_timeseries,(rad_test[i,:]*rad_test_stats[i,1] + rad_test_stats[i,0]), alpha=0.6,color='b',label = 'Actual')
print('Test_1:',rad_test_stats[i,0], rad_test_stats[i,1], np.mean(pred[0]), np.std(pred[0]))
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Transit Radius [$\rm R_{Jup}$]')
plt.legend(loc='upper right')

plt.subplot(2,2,2)
plt.title(f'Test Planet {i+2}')
pred=predict_combined(model_shape, model_mean, model_std, X_test[i+1:i+2])[0]
plt.plot(wl_timeseries,pred[0],color='r',label='Prediction')
plt.plot(wl_timeseries,(rad_test[i+1,:]*rad_test_stats[i+1,1] + rad_test_stats[i+1,0]), alpha=0.6,color='b',label = 'Actual')
print('Test_2:',rad_test_stats[i+1,0], rad_test_stats[i+1,1], np.mean(pred[0]), np.std(pred[0]))
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Transit Radius [$\rm R_{Jup}$]')
plt.legend(loc='upper right')

plt.subplot(2,2,3)
plt.title(f'Test Planet {i+3}')
pred=predict_combined(model_shape, model_mean, model_std, X_test[i+2:i+3])[0]
plt.plot(wl_timeseries,pred[0],color='r', label='Prediction')
plt.plot(wl_timeseries,(rad_test[i+2,:]*rad_test_stats[i+2,1] + rad_test_stats[i+2,0]), alpha=0.6,color='b',label = 'Actual')
print('Test_3:',rad_test_stats[i+2,0], rad_test_stats[i+2,1], np.mean(pred[0]), np.std(pred[0]))
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Transit Radius [$\rm R_{Jup}$]')
plt.legend(loc = 'upper right')

plt.subplot(2,2,4)
plt.title('WASP-121 b')
plt.plot(wl_timeseries,WASP_predict[0], color='r',label = 'Prediction')
plt.plot(wl_timeseries,real_radius_initial[0,:], alpha=0.6, color='b', label='Actual')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'Transit Radius [$\rm R_{Jup}$]')
plt.legend(loc='upper right')

plt.subplots_adjust(hspace=0.4)
plt.show()



plt.title("WASP-121 b Transmission Spectrum")
plt.plot(wl_timeseries,real_radius_initial[0,:], label='WASP 121 b Transmission Spectrum ')
plt.xlabel('Wavelength (m)')
plt.ylabel('Jupiter Radii')
plt.legend()
plt.show()


