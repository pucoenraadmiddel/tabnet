from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.metrics import mean_squared_error
import wandb

import pandas as pd
import numpy as np
np.random.seed(43)
import random
import argparse

print('Reading the data...')
train = pd.read_parquet(r'/home/coenraadmiddel/Documents/RossmannStoreSales/TabNet/tabnet/train_processed.parquet')
print("Read:", train.shape)


#select only a couple of columns

train = train[['Store',
                'DayOfWeek',
                'Promo',
                'StateHoliday',
                'SchoolHoliday',
                'StoreType',
                'Assortment',
                'CompetitionDistance',
                'Promo2SinceWeek',
                'Promo2SinceYear',
                'Year',
                'Month',
                'Day',
                'WeekOfYear',
                'CompetitionOpen',
                'PromoOpen',
                'IsPromoMonth',
                'Sales',
                'Set']]


if "Set" not in train.columns:
    train.reset_index(inplace=True, drop=True)
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index

X_all, y_all = train.drop(columns = ['Sales', 'Set']), np.log1p(train[['Sales']].values)

X_train = X_all.values[train_indices]
y_train = y_all[train_indices].reshape(-1, 1)

X_valid = X_all.values[valid_indices]
y_valid = y_all[valid_indices].reshape(-1, 1)

X_test = X_all.values[test_indices]
y_test = y_all[test_indices].reshape(-1, 1)
    
categorical_columns = ['Store',
                        'DayOfWeek',
                        'Promo',
                        'StateHoliday',
                        'SchoolHoliday',
                        'StoreType',
                        'Assortment',
                        # 'Year',
                        # 'Month',
                        # 'Day',
                        # 'WeekOfYear',
                        'IsPromoMonth']
#force categorical columns to the categorical type

train[categorical_columns] = train[categorical_columns].astype('category')

#get the indices of the categorical columns in train dataFrame

cat_idxs = [train.columns.get_loc(c) for c in categorical_columns if c in train]

#get the dimensions of the categorical columns in train dataFrame

cat_dims = [len(train[c].cat.categories) for c in categorical_columns if c in train]

y_dim = 1

X_all_cats = X_all[categorical_columns]
# make a list of the number of unique values in each categorical column

catmaxlist = [X_all_cats[col].nunique() for col in X_all_cats.columns]

# define your embedding sizes : here just a random choice
cat_emb_dim =  [random.randint(2, min(x, 10)) for x in catmaxlist]
print(cat_emb_dim)
print("Categorical Dimensions: ", len(cat_dims))
print("Categorical Embedding Dimensions: ", len(cat_emb_dim))
assert len(cat_dims) == len(cat_emb_dim)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using {}".format(DEVICE))

def main(opt):
    wandb.init(project="tabnet_all_rossmann", config=opt)
    opt = wandb.config
    
    print(opt)

    clf = TabNetRegressor(n_d=opt.n_d,
                          n_a=opt.n_a,
                          n_steps=opt.n_steps,
                          optimizer_params=dict(lr=opt.lr),
                          scheduler_params={"factor": opt.factor, "patience": opt.patience},
                        #   scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                            cat_idxs=cat_idxs,
                            cat_dims=cat_dims,  
                            verbose=1,     
                            device_name = DEVICE
                                   
                          
    )
    
    clf.fit(X_train=X_all.values[train_indices],
            y_train=y_all[train_indices],
             eval_set=[
            (X_train, y_train), 
            (X_valid, y_valid),
            (X_test, y_test)],
            eval_name=['train', 'valid', 'test'],
            eval_metric=['mse', 'rmse'],
            max_epochs=opt.max_epochs,
            batch_size=opt.batch_size,
            virtual_batch_size=opt.virtual_batch_size,)
    
    
    saving_path_name = "./tabnet_model"
    saved_filepath = clf.save_model(saving_path_name)
    
    # else:
    loaded_preds = clf.predict(X_test)
    loaded_test_mse = mean_squared_error(loaded_preds, y_test, squared=False)

    print(f"FINAL TEST SCORE FOR Rossmann : {loaded_test_mse}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_d', default=64, type=int)
    parser.add_argument('--n_a', default=64, type=int)
    parser.add_argument('--n_steps', default=5, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=10, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--virtual_batch_size', default=128, type=int)
    
    opt = parser.parse_args()
    
    main(opt)
    
    
