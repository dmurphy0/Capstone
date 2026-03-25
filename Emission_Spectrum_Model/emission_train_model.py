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


plt.rcParams.update({  #change fontsizes for presentation
    'font.size': 22,          # Global font size
    'axes.labelsize': 24,     # X and Y label size
    'axes.titlesize': 24,     # Title size
    'xtick.labelsize': 22,    # X-axis tick numbers
    'ytick.labelsize': 22,    # Y-axis tick numbers
    'legend.fontsize': 18     # Legend text
})


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #defines device as CPU or GPU.
startTime = datetime.now()


with h5py.File("emission_w_0.3_5.h5", "r") as f: #calls the wavelength distributon for plotting spectra
    wl_timeseries=f['emission_wl_0.3_5'][:]
wl_timeseries=wl_timeseries*1e4


with h5py.File("Emission_RQMC_combined.h5", "r") as f:  #calls data with RQMC sampling for training (16384 samples)
    X_temp = f['emission_input'][:]
    y_temp = f['emission_flux'][:]


with h5py.File("real_emission.h5","r") as f:  #calls data for WASP-121b for testing
    X_planet=f['real_emission_input'][:]
    y_planet=f['real_emission_flux'][:]


cols_to_log = [0,1,7]


X_planet [cols_to_log] = np.log(X_planet[cols_to_log])  #log transform the features that should be log distributed in data-set 
X_temp [:,cols_to_log] = np.log(X_temp[:, cols_to_log])


y_temp = y_temp.reshape(16384,3843) #need to train in log normal space
y_temp=y_temp[:,0:2814]
y_copy = y_temp.copy() #linear-space fluxes
y_temp=np.log(y_temp)  #log-space fluxes

y_planet = (y_planet.reshape(3843))  
y_planet=y_planet[0:2814] #cut down to 0.3 to 5 microns

flux_shape=y_temp


#preprocess data using sklearn
shape_scaler = StandardScaler()
flux_shape = shape_scaler.fit_transform(flux_shape)

preprocessor=MinMaxScaler() 
X_temp=preprocessor.fit_transform(X_temp)   #normalises input features. RQMC isnt normally distributed by design, so can only really use MinMax Scaling. 


#split data into training and validation data-sets
X_train, X_val, flux_shape_train, flux_shape_val = train_test_split(
    X_temp,  
    flux_shape,   
    test_size=0.2,
    random_state=42
)

################################################################

class LinearBlock(nn.Module):  #define linear blocks
    
    def __init__(self, in_features, out_features, dropout_rate=0.0, activation='leaky_relu'):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        # Batch Norm is typically applied before activation
        layers.append(nn.BatchNorm1d(out_features))

        if activation == 'gelu':   #only use GELU here, but other activation functions are from previous iterations 
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


class ResBlock(nn.Module):  #define residual blocks
    def __init__(self, features, dropout_rate=0.0):  
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


class Shape_Model(nn.Module): #construct model from linear and residual blocks
    def __init__(self):
        super().__init__()       
        
        self.input_layer = LinearBlock(9, 512, dropout_rate=0.0, activation='gelu')
        self.hidden = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            LinearBlock(512,1024, dropout_rate=0.0, activation='gelu'),   
        )    
        self.output_layer = nn.Linear(1024, 2814)


    def forward(self,x):
        x = self.input_layer(x)
        x = self.hidden(x)
        return self.output_layer(x) #standard feedforward architecture. No residual connections here as it wouldnt be helpful with learning. We are trying to approximate a complex shape. 
      
val_loss_plot=[] #use for epoch loss plots
train_loss_plot=[]

#train the model:
def train_single_model(model, train_loader, val_loader, criterion, epochs, patience, model_name="Model"):   
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)   #ADAMW optimiser
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  #decrease learning rate after 3 epochs with no val_loss decrease. Pushes further accuracy
    
    # Simple Early Stopping Logic Variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'loss': [], 'val_loss': []}

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)  
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        

        # --- VALIDATE ---
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_loss = criterion(preds, targets)
                val_running_loss += val_loss.item() * inputs.size(0)
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        
        #scheduler step
        scheduler.step(epoch_val_loss)

        # Record History
        history['loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)

        #plot epoch loss data
        val_loss_plot.append(epoch_val_loss)
        train_loss_plot.append(epoch_loss)

        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        # --- EARLY STOPPING CHECK ---
        if epoch_val_loss < best_loss:
            best_loss=epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
      
    # Load best weights from training before returning
    model.load_state_dict(best_model_wts)
    
    return model, history

################################################################

train_dataset_shape = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), 
    torch.tensor(flux_shape_train, dtype=torch.float32)
)
val_dataset_shape = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32), 
    torch.tensor(flux_shape_val, dtype=torch.float32)
)


#1 Shape
train_loader_shape = DataLoader(train_dataset_shape, batch_size=256, shuffle=True)
val_loader_shape = DataLoader(val_dataset_shape, batch_size=256, shuffle=False)   #set bacth_size at 256. Smaller batch size makes process more stochastic. 

model_shape = Shape_Model().to(device)  

class SpectralSmoothnessLoss(nn.Module):  #custom loss function to maximise accuracy 
    def __init__(self, alpha=2, beta=1, delta=0.1):    
        super().__init__()
        self.alpha = alpha #Strength of the smoothness penalty
        self.mse = nn.MSELoss()
        self.delta=delta #HuberLoss parameter
        self.huber= nn.HuberLoss(delta=self.delta)
        self.beta=beta #Strength of curvature penalty
        
    def forward(self, pred, target):    
        #1. Weighted Huber Loss      
        base_huber = self.huber(pred, target)
        weights = torch.exp(target) #Weighs larger fluxes higher for lin-space accuracy 
        weights = weights / torch.mean(weights)
        weighted_huber = torch.mean(base_huber * weights)

        #2. Smoothness (First Derivative) Penalty
        diff1_pred = pred[:, 1:] - pred[:, :-1]
        diff1_target = target[:, 1:] - target[:, :-1]
        loss_smooth = self.huber(diff1_pred, diff1_target)
       
        #3. Second Derivative Penalty (Curvature matching)
        diff2_pred = pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2]
        diff2_target = target[:, 2:] - 2*target[:, 1:-1] + target[:, :-2]
        loss_curvature = self.huber(diff2_pred, diff2_target)
        
        # Total Loss
        return weighted_huber + (self.alpha * loss_smooth) + (self.beta * loss_curvature)

# Instantiate Model
print("Training Shape Model...")
model_shape, hist_shape = train_single_model(
    model_shape, train_loader_shape, val_loader_shape, 
    criterion=SpectralSmoothnessLoss(), epochs=70, patience=10, model_name="Shape"   #set maximum number of epochs, and early stopping patience
)

print('Total Duration: {}'.format(datetime.now() - startTime))

torch.save(model_shape.state_dict(), 'emis_shape_11.pth')  #saves model

plt.plot(np.linspace(1,len(val_loss_plot), len(val_loss_plot)), val_loss_plot, label = 'Val_Loss', color='b', alpha=0.6)
plt.plot(np.linspace(1,len(train_loss_plot), len(train_loss_plot)), train_loss_plot, label='Train_Loss', color = 'r')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (Standard Deviations)")
plt.show()

#prediction function
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

