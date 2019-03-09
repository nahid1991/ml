# These are for exploring data
import pandas as pd
import researchpy as rp
import matplotlib.pyplot as plt

# These are for running the model and conducting model diagnostics
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip

df = pd.read_csv('insurance.csv')

print("============================================")
# Let's get more information on the continuous varibles
print(rp.summary_cont(df[['charges', 'age', 'children']]))

print("\n===========================================")
# Let's get more information on the categorical data
print(rp.summary_cat(df[['sex', 'smoker', 'region']]))

df['sex'].replace({'female' : 1, 'male' : 0}, inplace= True)
df['smoker'].replace({'no': 0, 'yes': 1}, inplace= True)

df = pd.get_dummies(df)

print("\n===========================================")
print(df.head())

print("\n===========================================")
model = smf.ols(
    "charges ~ age + bmi + sex + smoker + children + region_northwest + region_southeast + region_southwest",
    data=df).fit()

print(model.summary())
print("\n===========================================")

print(df.corr())

stats.probplot(model.resid, plot=plt)
plt.title("Model1 Residuals Probability Plot")
plt.show()

print("\n===========================================")
print(stats.kstest(model.resid, 'norm'))

name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model.resid, model.model.exog)
print(lzip(name, test))

print("\n===========================================")
sex_variance_results = stats.levene(df['charges'][df['sex'] == 0],
                                    df['charges'][df['sex'] == 1], center='median')

smoker_variance_results = stats.levene(df['charges'][df['smoker'] == 0],
                                       df['charges'][df['smoker'] == 1], center='median')

region_variance_results = stats.levene(df['charges'][df['region_northeast'] == 1],
                                       df['charges'][df['region_northwest'] == 1],
                                       df['charges'][df['region_southeast'] == 1],
                                       df['charges'][df['region_southwest'] == 1], center='median')

print(f"Sex Variance: {sex_variance_results}", "\n",
      f"Smoker Variance: {smoker_variance_results}", "\n",
      f"Region Variance: {region_variance_results}", "\n")


print("\n===========================================")
model3 = smf.ols(
    "charges ~ age + bmi + sex + smoker + children + region_northwest + region_southeast + region_southwest",
    data=df).fit(cov_type='HC3')

print(model3.summary())
