

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Previsão de renda", layout="wide")


# colocando os caminhos para o df original e o meu modelo

df = pd.read_csv(r"C:\Users\felip\OneDrive\Ebac\Desenvolvimento Modelos com Pandas e Py\2º Projeto\Profissão Cientista de Dados M16 2 Projeto\projeto 2\input\previsao_de_renda.csv")
MODEL_PATH = Path("./modelo_renda.pkl") 


st.title("Previsão de renda")
st.caption("Modelo e regressão que preve renda com base em caracteristicas")
st.write(
    "A ideia aqui é explorar cenários de renda, "
    "usando ferramentas do streamlit."
)
st.divider()


# X = df[["sexo_M", "posse_de_imovel_True", "idade", "tempo_emprego"]]

# aqui vou colocar as opções para o usuário

sexo = st.sidebar.selectbox("Sexo", df["sexo"].unique())
sexo_M = (sexo == "M")       

posse_de_imovel_True = st.sidebar.checkbox("Tem imóvel?")

idade = st.sidebar.slider(
    "Idade",
    min_value=int(df["idade"].min()),
    max_value=int(df["idade"].max()),
    value=int(df["idade"].median()),
    step=1
)

tempo_emprego = st.sidebar.slider(
    "Tempo de trabalho em anos",
    min_value=int(df["tempo_emprego"].min()),
    max_value=int(df["tempo_emprego"].max()),
    value=int(df["tempo_emprego"].median()),
    step=1  
)


#Dados para alimentar o meu modelo
entrada = pd.DataFrame([{
    "sexo_M": sexo_M,                          
    "posse_de_imovel_True": posse_de_imovel_True,  
    "idade": float(idade),
    "tempo_emprego": float(tempo_emprego)
}])


@st.cache_resource
def carregar_modelo(caminho: Path):
    with open(caminho, "rb") as f:
        return pickle.load(f)
    

modelo_carregado = carregar_modelo(MODEL_PATH)



pred_log = modelo_carregado.predict(entrada)[0]
renda_prevista = float(np.exp(pred_log))
st.subheader("Resultado da previsão")
st.metric(label="Renda estimada", value=f"R$ {renda_prevista:,.2f}")




# Nova análise

import seaborn as sns
import matplotlib.pyplot as plt

st.header("Análise Exploratória da Renda")

mostrar_eda = st.checkbox("Mostrar análise exploratória da base de renda")

if mostrar_eda:
    df_eda = df.sample(min(5000, len(df)), random_state=42)

    aba = st.tabs(["Distribuições", "Renda x Variáveis", "Correlação"])


    with aba[0]:
        st.subheader("Distribuição da Renda")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_eda["renda"], kde=False, bins=30, ax=ax)
        st.pyplot(fig)

        st.subheader("Distribuição da Idade")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_eda["idade"], kde=False, bins=30, ax=ax)
        st.pyplot(fig)

    with aba[1]:
        st.subheader("Renda média por Sexo")
        renda_por_sexo = df.groupby("sexo")["renda"].mean().reset_index()
        st.bar_chart(renda_por_sexo, x="sexo", y="renda")

        st.subheader("Renda por Posse de Imóvel")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_eda, x="posse_de_imovel", y="renda", ax=ax)
        st.pyplot(fig)

        st.subheader("Idade vs Renda")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            x=df_eda["idade"],
            y=np.log(df_eda["renda"]),
            alpha=0.3,
            ax=ax
        )
        st.pyplot(fig)

    with aba[2]:
        st.subheader("Correlação")
        corr = df_eda[["idade", "tempo_emprego", "renda"]].corr()

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
