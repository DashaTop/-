import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sample_submission = pd.read_csv("./sample_submission.csv")
test = pd.read_csv("./test.csv")
train = pd.read_csv("./train.csv")





# Преобразование Nane (NA)
# Alley: NA на No
# print(train.info())
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
# BedroomAbvGr: NA на 0
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual: Na na No or 0
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir Na na No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition Na na Norm
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
# External stuff : NA na TA
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
# Fence : Na na No
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
# FireplaceQu : Na na No or 0
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
# Functional : Na na Typ
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
# GarageType etc : NA na No or 0
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA na 0
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA na TA
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA na 0
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA na TA
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA na 0
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA na Reg
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA na None or 0
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : NA na No or 0
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA na 0
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA na N
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
# PoolQC : NA na No or 0
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA na normal
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA na 0
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA na 0
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA na AllPub
train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA na 0
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
# Electrical : NA na FuseA
train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna("FuseA")

# График соотношения жилой площади и стоимости
# plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
# plt.title("GrLivArea-SalePrice")
# plt.xlabel("GrLivArea")
# plt.ylabel("SalePrice")
# plt.show()
train = train[train.GrLivArea < 4000]

# График разброса цен
# plt.plot(train.SalePrice)
# plt.show()


ans = np.log(train.SalePrice.to_frame())
features = train.drop(["Id","SalePrice"], 1).copy()


# перевод категорийных признаков в Числоые
features[features.keys()] = features[features.keys()].apply(LabelEncoder().fit_transform)


scalar = StandardScaler()

features[features.keys()] = scalar.fit_transform(features[features.keys()])

print(features)


kfold = KFold(n_splits=10, shuffle=True)
regs = []
random_state = 0
regs.append(Lasso())
regs.append(ElasticNetCV())
regs.append(SVR())
regs.append(KNeighborsRegressor())
regs.append(DecisionTreeRegressor(random_state=random_state))
regs.append(RandomForestRegressor(random_state=random_state))
regs.append(GradientBoostingRegressor(max_features='sqrt'))

results = []
for reg in tqdm(regs):
    results.append(np.sqrt(-cross_val_score(reg, features, y=ans.values.ravel(), scoring="neg_mean_squared_error", cv=kfold)))


means = []
errors = []
for result in results:
    means.append(result.mean())
    errors.append(result.std())
res_frame = pd.DataFrame({"CrossValMeans": means, "CrossValerrors": errors, "Algorithm": ["LinearRegression",
                                                                                          "ElasticNetCV",
                                                                                          'SVR',
                                                                                          'KNeighborsRegressor',
                                                                                          'DecisionTreeRegressor',
                                                                                          'RandomForestRegressor',
                                                                                          'GradientBoostingRegressor']})

print(res_frame)
g = sns.barplot("CrossValMeans", "Algorithm", data=res_frame, palette="Set3", orient="h", **{'xerr': errors})
g.set_xlabel("Mean Squared Error")
g = g.set_title("Cross validation scores")
plt.show()