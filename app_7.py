import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.impute import KNNImputer
from numpy import NaN
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder



st.title('Prédiction de scoring bancaire')

st.write('''
Cette application permet de prédire l'attribution de crédit bancaire selon différentes caractéristiques du client
''')

#DATASET
@st.cache
def load_data():
    path_df =("x_sample.csv")
    df = pd.read_csv(path_df,nrows=1000,index_col='SK_ID_CURR', encoding ='utf-8')#,index_col='SK_ID_CURR',nrows=100
    return df

df = load_data()
df=df.iloc[:,1:]


# CLIENT
def client(df, id):
        client =df[df.index == int(id)]# df[df.SK_ID_CURR == int(id)]#df[df.index == int(id)]
        return client



#MODEL
def load_model():
        '''loading the trained model'''
        inpt=open('best_model','rb')
        model=pickle.load(inpt)
        return model
model=load_model()




# SIDBAR
st.sidebar.header("Paramètres d'entrée")

 #Loading selectbox
ID= st.sidebar.selectbox("Client ID", df.index)

infos_client = client(df, ID)
st.sidebar.markdown("**Gender : **",)
st.sidebar.text( infos_client["CODE_GENDER"].values[0])

st.sidebar.markdown("**Age : **")
st.sidebar.text("{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/-365)))

st.sidebar.markdown("**Family status : **")
st.sidebar.text(infos_client["NAME_FAMILY_STATUS"].values[0])


st.sidebar.markdown("**Number of children : **")
st.sidebar.text("{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))




# HOME PAGE

st.write("**Client selectionné: **",ID)

#positionnement du client =>Age distribution plot
@st.cache
def load_age_population(df):
    data_age = round((df["DAYS_BIRTH"]/-365), 2)
    return data_age


@st.cache
def load_EXT_SOURCE_3_population(df):
    data_EXT_SOURCE_3 = round(df["EXT_SOURCE_3"], 2)
    return data_EXT_SOURCE_3


data_age = load_age_population(df)
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data_age, edgecolor = 'k', color="y", bins=20)
ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle='--')
ax.set(title='Age du client', xlabel='Age(ans)')
st.pyplot(fig)

#positionnement du client =>income distribution plot
@st.cache
def load_income_population(df):
    df_income = pd.DataFrame(df["AMT_INCOME_TOTAL"])
    return df_income

data_income = load_income_population(df)
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data_income, edgecolor = 'k', color="y", bins=20)
ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="red", linestyle='--')
ax.set(title='Revenu du client', xlabel='Revenus')
st.pyplot(fig)


#pretraitement des données brutes

# 1-encodage
encoder = LabelEncoder()
count = 0
# Iterate through the columns
for col in df:
    # If 2 or fewer unique categories
    if df.loc[:,col].dtype == 'object' and len(list(df.loc[:,col].unique())) <= 2:
        # Train on the training data
        encoder.fit(df.loc[:,col])
        # Transform both training and testing data
        df.loc[:,col] = encoder.transform(df.loc[:,col])

        count += 1
df1 = pd.get_dummies(df)


# 2- Imputation
imputer = SimpleImputer(strategy='mean')
DF= imputer.fit_transform(df1).tolist()#.values.reshape(-1,1)
DF2=pd.DataFrame(DF, columns=df1.columns, index=df.index)

#3-Normalisation
scaler= MinMaxScaler()

DFF= scaler.fit_transform(DF2).tolist()#.values.reshape(-1,1)
DF3= pd.DataFrame(DFF,columns=DF2.columns,index=df.index)#, columns=DF.columns)

#st.write(DF3.head(2))

# Fonction de Prédiction
@st.cache
def load_prediction(df, model):
    score =( model.predict_proba(df)[:,1])>= 0.44
    return score


# Prédictions
pred=load_prediction(DF3,model).tolist()

df['accord_crédit']=pred


# DECISION D'ATTRIBUTION DE CREDIT
st.subheader("Attribution du crédit")
if st.checkbox("Le client est-il Solvable ?"):
    p=model.predict_proba(DF3)[:,1].tolist()
    if pred[pred.index==ID]=='True':
        st.write("OUI")
        st.write("Le client a une probabilité de solvabilité est de :",round(p[p.index==ID],2))
    else:
        st.write("NON")
        st.write("Le client a une probabilité de solvabilité est de :",round(p[p.index==ID],2))

#position du client en terme de probabilité d'attribution du crédit
        fig, ax = plt.subplots(figsize=(10, 5))
        p=model.predict_proba(DF3)[:,1].tolist()
        sns.histplot(p, edgecolor = 'k', color="y", bins=20)
        ax.axvline(p[p.index==ID], color="red", linestyle='--')
        ax.set(title='Situation du client', xlabel='probabilité d attribution du crédit')
        st.pyplot(fig)

else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)


# Ajout de la prédiction au dataframe

encoder= LabelEncoder()
df['accord_crédit']=encoder.fit_transform(df['accord_crédit'])



st.subheader("Position du client")
if st.checkbox("Voir la situation du client sélectionné par rapport aux autres?"):


# COMPARAISON Du Score normalisé à partir d'une source de données externe DU CLIENT PAR RAPPORT AU AUTRES PAYEURS
    fig, ax = plt.subplots(figsize=(10, 5))
    P=df["accord_crédit"].tolist()
    sns.kdeplot(df[df['accord_crédit'] == 0]['EXT_SOURCE_3'], label = 'client non solvable')
    sns.kdeplot(df[df['accord_crédit'] == 1]['EXT_SOURCE_3'], label = 'client solvable')
    ax.axvline(infos_client["EXT_SOURCE_3"].values, color="red", linestyle='--')
    ax.set(xlabel='Score normalisé à partir d une source de données externe',
    ylabel='Density', title='Score normalisé à partir dune source de données externe')
    plt.legend()
    st.pyplot(fig)





# COMPARAISON DE L4AGE DU CLIENT PAR RAPPORT AU AUTRES PAYEURS
    fig, ax = plt.subplots(figsize=(10, 5))
    P=df["accord_crédit"].tolist()
    sns.kdeplot(df[df['accord_crédit'] == 0]['DAYS_BIRTH']/-365, label = 'client non solvable')
    sns.kdeplot(df[df['accord_crédit'] == 1]['DAYS_BIRTH']/-365, label = 'client solvable')
    ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle='--')
    ax.set(xlabel='Age(ans)',ylabel='Density', title='Distribution de l age')
    plt.legend()
    st.pyplot(fig)



# comparaison du revenu du client par rapport au autres demandeurs de accord_crédit
    fig, ax = plt.subplots(figsize=(10, 5))
    P=df["accord_crédit"].tolist()
    sns.kdeplot(df[df['accord_crédit'] == 0]["AMT_INCOME_TOTAL"], label = 'client non solvable')
    sns.kdeplot(df[df['accord_crédit'] == 1]["AMT_INCOME_TOTAL"], label = 'client solvable')
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"]), color="red", linestyle='--')
    ax.set(xlabel='Revenus', ylabel='Density', title='Distribution des revenus')
    plt.legend()
    st.pyplot(fig)


else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)





#
