import numpy as np
import pandas as pd
import pickle, sys
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
from scipy.stats import uniform, randint

# Define functions for loading data
def get_params(file):
    """
    This function reads in the simulation parameters for all DREAMS sims and converts them into physical units
    
    Inputs
     - file - absolute or relative path to the file containing all simulation parameters
     
    Returns
     - params - an Nx4 ndarray with the following parameters
                index 0 - WDM particle mass [keV]
                index 1 - SN1 (ew) wind energy
                index 2 - SN2 (kw) wind velocity
                index 3 - AGN1 (BHFF) black hole feedback factor
    """
    sample = np.loadtxt(file)
    
    fiducial = np.array([1,3.6,7.4,.1]) #fiducial TNG values
    params = sample * fiducial
    params[:,0] = 1/params[:,0]
    #ask Jonah about
    
    return params

def norm_params(params):
    """
    This function normalizes the four simulation parameters (WDN, SN1, SN2, AGN).
    
    Inputs
     - params - an Nx4 array of simulation parameters
    
    Results
     - nparams - same as the input but now normalized and linearly sampled between 0 and 1
    """
    nparams = params / np.array([1, 3.6, 7.4, .1])
    nparams[:,0] = 1/nparams[:,0]
    
    nparams[:,1:] = np.log10(nparams[:,1:])

    minimum = np.array([1/30, np.log10(0.25), np.log10(0.5), np.log10(0.25)])
    maximum = np.array([1/1.8, np.log10(4.0), np.log10(2.0), np.log10(4.0)])

    nparams = (nparams - minimum)/(maximum - minimum)

    return nparams

with open(sys.argv[1], "rb") as f:
    catalogs = pickle.load(f)

# Getting parameters

param_path = sys.argv[2]
params = []
boxes = range(1024)
for box in boxes:
    try:
        param = get_params(param_path)[box]
        params.append(param)
    except:
        print(box)
params = np.array(params)
nparams = norm_params(params)
params = nparams[:,0:1]

# Filtering data and creating features
n_subs,MgMd,Mg,Md,Ms,SFR,snap,MgMs,MsMd = [],[],[],[],[],[],[],[],[]

for cat in catalogs:
    subhalo_mass_type = np.array(cat["SubhaloMassType"])
    dm_mass = subhalo_mass_type[:, 1] * 1e10
    snapnum = np.array(cat["SnapNum"])
    sfr = np.array(cat['SubhaloSFR'])
    
    mask = (10**7.5 < dm_mass) & (dm_mass < 10**8.5) & (snapnum > 11)
    n_subs.append(np.count_nonzero(mask))
    
    M_gas = subhalo_mass_type[:, 0][mask]
    M_DM  = subhalo_mass_type[:, 1][mask]
    M_stellar = subhalo_mass_type[:,4][mask]
    sfr=sfr[mask]
    snapnum=snapnum[mask]

    mask = M_DM > 0
        
    ratio1 = M_gas[mask] / M_DM[mask]  
    p84_ratio1 = np.percentile(ratio1, 84)
    
    ratio2 = M_gas[mask] / M_stellar[mask]  
    p84_ratio2 = np.percentile(ratio1, 84)

    ratio3 = M_stellar[mask] / M_DM[mask]  
    p84_ratio3 = np.percentile(ratio1, 84)
    
    p84_mg = np.percentile(M_gas, 84)
    p84_md = np.percentile(M_DM, 84)
    p84_ms = np.percentile(M_stellar, 84)
    p84_sfr = np.percentile(sfr, 84)
    p84_snap = np.percentile(snapnum, 84)
    
    MgMd.append(p84_ratio1)
    MgMs.append(p84_ratio2)
    MsMd.append(p84_ratio3)
    Mg.append(p84_mg)
    Md.append(p84_md)
    Ms.append(p84_ms)
    SFR.append(p84_sfr)
    snap.append(p84_snap)

y = params.flatten()
X = pd.DataFrame({'Nodes':n_subs,'M_gas':Mg,
                  'M_dm':Md,'M_s':Ms,'SFR':SFR,'Mg/Md':MgMd,'Snap':snap,'Mg/Ms':MgMs,'Ms/Md':MsMd})

# Splitting dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Define parameter distributions
param_dist = {
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.2),      
    "subsample": uniform(0.6, 0.4),           
    "colsample_bytree": uniform(0.6, 0.4),    
    "min_child_weight": uniform(1, 9),        
    "n_estimators": randint(100, 600)
}

# Randomized search
search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,                     
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit search
search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", search.best_params_)

# Best model
best_xgb = search.best_estimator_
# Evaluate
y_pred = best_xgb.predict(X_test)
R_sq = r2_score(y_test,y_pred)
print(R_sq)

# Plotting feature importance
importance = best_xgb.get_booster().get_score(importance_type="gain")

# Normalize to sum = 1
total = sum(importance.values())
norm_importance = {k: v / total for k, v in importance.items()}

# Convert to DataFrame for plotting
df = pd.DataFrame(list(norm_importance.items()), columns=["Feature", "Importance"])
df = df.sort_values("Importance", ascending=True)

# Plot
plt.figure(figsize=(9, 6))
plt.barh(df["Feature"], df["Importance"])
plt.xlabel("Fraction of total importance")
plt.savefig("xgboost.png")
