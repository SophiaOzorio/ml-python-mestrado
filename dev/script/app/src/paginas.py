import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load
from scipy.stats import chi2_contingency, ks_2samp, levene, mannwhitneyu, spearmanr
from scipy.stats import ttest_ind
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# Só o Clustering é importado: classifier.py/regression.py re-treinam o
# GridSearch (e o regression.py importa o TensorFlow), então o app lê os
# resultados que os notebooks de treinamento já salvaram.
from dev.script.src.modelos.clustering import Clustering

# O Streamlit roda sem tela; evita plt.show() tentar abrir janela
plt.switch_backend("Agg")

# ----------------------------------------------------------------------
# Configurações do projeto (ajuste aqui, o resto do código não muda)
# ----------------------------------------------------------------------
ALVO_CLF = "ataxia"
ALVO_REG = "idade_inicio_sintomas"

# Código do alvo (LabelEncoder) -> nome original da classe
NOMES_CLASSES: dict = {0: "cerebelar", 1: "esporadica", 2: "nao especificada"}

# Variáveis contínuas usadas no clustering (as mesmas do notebook)
COLUNAS_NUM = [
    "cadd_phred",
    "conservacao_phylop",
    "profundidade_leitura",
    "gnomad_af_log",
    "idade_inicio_sintomas",
]

MODELOS_CLUSTER = {
    "kmeans": "K-means",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative",
}

# Nomes das colunas de predição salvas nos notebooks
MODELOS_CLF = ["k-NN", "SVM", "RF", "GBM", "NB", "NN"]
MODELOS_REG = ["Regressão Linear Múltipla", "SVR", "Bayesian Ridge", "Rede Neural"]

CONJUNTOS = {"Treino": "train", "Validação": "valid", "Teste": "test"}

#Procura, subindo a partir deste arquivo, a pasta que contém "results"
# (dev/script). Assim funciona esteja o paginas.py em dev/script/src ou em
# dev/script/app/src, e não depende do diretório de execução.
def _acha_dir_script() -> Path:
    for pasta in Path(__file__).resolve().parents:
        if (pasta / "results").is_dir():
            return pasta
    return Path(__file__).resolve().parents[1]
 
 
DIR_SCRIPT = _acha_dir_script()
DIR_SPLIT = DIR_SCRIPT / "results" / "split_dataset"
DIR_CLASSIF = DIR_SCRIPT / "results" / "classificacao"
# Resultados da regressão: aceita results/regressao (organizado) ou
# results/split_dataset (onde o notebook salvou); usa a primeira que estiver completa
DIR_REGR = [
    DIR_SCRIPT / "results" / "regressao",
    DIR_SCRIPT / "results" / "split_dataset",
]
 
# ----------------------------------------------------------------------
# Busca de dados (com cache)
# ----------------------------------------------------------------------
def _le_split(conjunto: str) -> pd.DataFrame:
    """
    Junta X + y_idade + y_ataxia, como nos notebooks de treinamento.
    """
    X = pd.read_csv(DIR_SPLIT / f"X_{conjunto}_ready.csv")
    partes = [X]
    if ALVO_REG not in X.columns:
        y_idade = pd.read_csv(DIR_SPLIT / f"y_idade_{conjunto}_ready.csv")
        partes.append(y_idade[[ALVO_REG]])
    partes.append(pd.read_csv(DIR_SPLIT / f"y_ataxia_{conjunto}_ready.csv"))
    return pd.concat(partes, axis=1)
 


@st.cache_data
def _busca_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna os DataFrames de treino, validação e teste.
    """
    return tuple(_le_split(c) for c in ("train", "valid", "test"))


@st.cache_data
def _busca_resultados(tipo: str) -> dict[str, pd.DataFrame] | None:
    """
    Carrega df_{train,valid,test}_resultados_{tipo} (joblib).
    tipo: 'classif' ou 'regression'. Retorna None se algum arquivo faltar.
    """
    pasta = DIR_CLASSIF if tipo == "classif" else DIR_REGR
    caminhos = {
        c: pasta / f"df_{c}_resultados_{tipo}" for c in ("train", "valid", "test")
    }
    if not all(p.is_file() for p in caminhos.values()):
        return None
    return {c: load(p) for c, p in caminhos.items()}


@st.cache_data(show_spinner="Executando clustering...")
def _roda_clustering(
    modelo: str,
    variavel_1: str,
    variavel_2: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[str, list]:
    """
    Executa a classe Clustering (sem alterá-la) e captura o que ela imprime
    e os gráficos que ela cria.
    """
    plt.close("all")
    cluster = Clustering(
        c1=variavel_1, c2=variavel_2, df=df_train, modelo=modelo, alvo=ALVO_CLF
    )

    saida = io.StringIO()
    with redirect_stdout(saida):
        # Grid search por validação cruzada + métricas + gráficos (treino)
        cluster.cross_validation(print_metricas=True)
        print()
        # Consistência dos clusters do treino no conjunto de teste
        cluster.testar_modelo(best_param=cluster.best_params, df_test=df_test)

    figuras = [plt.figure(n) for n in plt.get_fignums()]
    plt.close("all")
    return saida.getvalue(), figuras


# ----------------------------------------------------------------------
# Métricas, testes e gráficos
# ----------------------------------------------------------------------
def _metricas_clf(y_true, y_pred) -> dict:
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _metricas_reg(y_true, y_pred) -> dict:
    rho, p_rho = spearmanr(y_true, y_pred)
    return {
        "R Spearman": rho,
        "R p-value": p_rho,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def _teste_qui_quadrado(pred_treino, pred_teste) -> str:
    """
    Qui-quadrado: a distribuição das classes preditas no treino e no teste
    é homogênea?
    """
    try:
        tabela = pd.crosstab(
            np.repeat(["Treino", "Teste"], [len(pred_treino), len(pred_teste)]),
            np.concatenate([np.asarray(pred_treino), np.asarray(pred_teste)]),
        )
        chi2, p, gl, _ = chi2_contingency(tabela)
    except ValueError:
        return "Teste qui-quadrado não aplicável (poucas classes preditas)."

    if p < 0.05:
        conclusao = "DIFERENTE (heterogênea): rejeita H0"
    else:
        conclusao = "IGUAL (homogênea): falha em rejeitar H0"
    return (
        "Distribuição das classes preditas, treino x teste\n"
        f"χ² = {chi2:.4f} | gl = {gl} | p = {p:.4f}\n"
        f"Conclusão: {conclusao}"
    )


def _testes_regressao(pred_treino, pred_teste) -> pd.DataFrame:
    """
    Compara a distribuição das previsões do treino com a do teste.
    """
    ks, p_ks = ks_2samp(pred_treino, pred_teste)
    t, p_t = ttest_ind(pred_treino, pred_teste, equal_var=False)
    u, p_u = mannwhitneyu(pred_treino, pred_teste, alternative="two-sided")
    w, p_w = levene(pred_treino, pred_teste)

    linhas = [
        ("Kolmogorov-Smirnov (distribuição)", ks, p_ks),
        ("Teste T de Welch (médias)", t, p_t),
        ("Mann-Whitney (medianas)", u, p_u),
        ("Levene (variâncias)", w, p_w),
    ]
    return pd.DataFrame(
        {
            "Teste": [n for n, _, _ in linhas],
            "Estatística": [round(s, 4) for _, s, _ in linhas],
            "p-valor": [round(p, 4) for _, _, p in linhas],
            "Conclusão": [
                "Diferentes (rejeita H0)" if p < 0.05 else "Iguais (não rejeita H0)"
                for _, _, p in linhas
            ],
        }
    )


def _fig_confusao(y_true, y_pred, titulo: str):
    classes = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    rotulos = [NOMES_CLASSES.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=rotulos).plot(
        ax=ax, cmap=plt.cm.Greens, colorbar=False, xticks_rotation=45
    )
    ax.set_title(titulo)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fig_regressao(y_true, y_pred, titulo: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.regplot(x=y_true, y=y_pred, ax=ax, scatter_kws={"alpha": 0.5})
    ax.set(title=titulo, xlabel="Observado", ylabel="Predito")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ----------------------------------------------------------------------
# Páginas
# ----------------------------------------------------------------------
class Paginas:
    def _clustering(self) -> None:
        df_train, _, df_test = _busca_splits()
        variaveis = [c for c in COLUNAS_NUM if c in df_train.columns]
        if len(variaveis) < 2:
            st.warning("Não há variáveis contínuas suficientes para o clustering.")
            return None

        col = st.columns((2, 2, 4))
        with col[0]:
            variavel_1 = st.selectbox("Primeira variável:", variaveis, index=0)
        with col[1]:
            variavel_2 = st.selectbox("Segunda variável:", variaveis, index=1)
        with col[2]:
            modelo = st.radio(
                "Modelo:",
                list(MODELOS_CLUSTER),
                format_func=MODELOS_CLUSTER.get,
                horizontal=True,
            )

        if variavel_1 == variavel_2:
            st.warning("Escolha duas variáveis diferentes.")
            return None

        try:
            texto, figuras = _roda_clustering(
                modelo, variavel_1, variavel_2, df_train, df_test
            )
        except Exception as erro:  # mostra o erro em vez de derrubar o app
            st.error(f"Não foi possível executar o clustering: {erro}")
            return None

        st.code(texto, language=None)
        if figuras:
            col = st.columns(len(figuras))
            for c, fig in zip(col, figuras):
                with c:
                    st.pyplot(fig=fig)
        return None

    def _classificacao(self) -> None:
        res = _busca_resultados("classif")
        if res is None:
            st.info("Resultados da classificação não encontrados.")
            return None

        escolha = st.radio(
            "Conjunto avaliado:",
            list(CONJUNTOS),
            index=2,
            horizontal=True,
            key="conjunto_clf",
        )
        df = res[CONJUNTOS[escolha]]
        modelos = [m for m in MODELOS_CLF if m in df.columns]
        y_true = df[ALVO_CLF].to_numpy()

        tabela = pd.DataFrame(
            {m: _metricas_clf(y_true, df[m].to_numpy()) for m in modelos}
        ).T.round(4)
        st.dataframe(tabela)

        for ini in range(0, len(modelos), 3):
            col = st.columns((5, 0.5, 5, 0.5, 5))
            for pos, m in zip((0, 2, 4), modelos[ini : ini + 3]):
                with col[pos]:
                    acc = tabela.loc[m, "Acurácia"]
                    st.markdown(
                        f"<h4 style='text-align: center; '>{m}</h4>",
                        unsafe_allow_html=True,
                    )
                    st.pyplot(
                        fig=_fig_confusao(
                            y_true,
                            df[m].to_numpy(),
                            f"{escolha} - Acurácia {acc:.2f}",
                        )
                    )
                    st.text(_teste_qui_quadrado(res["train"][m], res["test"][m]))
            st.divider()
        return None

    def _regressao(self) -> None:
        res = _busca_resultados("regression")
        if res is None:
            st.info("Resultados da regressão não encontrados.")
            return None

        escolha = st.radio(
            "Conjunto avaliado:",
            list(CONJUNTOS),
            index=2,
            horizontal=True,
            key="conjunto_reg",
        )
        df = res[CONJUNTOS[escolha]]
        modelos = [m for m in MODELOS_REG if m in df.columns]
        y_true = df[ALVO_REG].to_numpy()

        tabela = pd.DataFrame(
            {m: _metricas_reg(y_true, df[m].to_numpy()) for m in modelos}
        ).T.round(4)
        st.dataframe(tabela)

        for ini in range(0, len(modelos), 2):
            col = st.columns((5, 0.5, 5))
            for pos, m in zip((0, 2), modelos[ini : ini + 2]):
                with col[pos]:
                    r2 = tabela.loc[m, "R2"]
                    st.markdown(
                        f"<h4 style='text-align: center; '>{m}</h4>",
                        unsafe_allow_html=True,
                    )
                    st.pyplot(
                        fig=_fig_regressao(
                            y_true, df[m].to_numpy(), f"{escolha} - R² {r2:.2f}"
                        )
                    )
                    st.caption("Previsões do treino x teste")
                    st.dataframe(
                        _testes_regressao(
                            res["train"][m].to_numpy(), res["test"][m].to_numpy()
                        ),
                        hide_index=True,
                    )
            st.divider()
        return None

    def _datasets(self) -> None:
        df_train, df_valid, df_test = _busca_splits()
        col = st.columns((2, 0.05, 2, 0.05, 2))
        for pos, titulo, df in (
            (0, "Dados de Treino", df_train),
            (2, "Dados de Validação", df_valid),
            (4, "Dados de Teste", df_test),
        ):
            with col[pos]:
                st.markdown(
                    f"<h4 style='text-align: center; '>{titulo}</h4>",
                    unsafe_allow_html=True,
                )
                st.dataframe(df)

        col = st.columns((2, 10))
        with col[0]:
            metr = pd.DataFrame(
                {
                    "Conjunto": ["Treino", "Validação", "Teste"],
                    "Tamanho": [len(df_train), len(df_valid), len(df_test)],
                }
            )
            st.write("Tamanho dos datasets:")
            st.dataframe(metr)
        return None

    def metricas(self) -> str:
        """
        Monta as abas do app.
        Retorna:
            str: Mensagem de sucesso.
        """
        abas = st.tabs(["Clustering", "Classificação", "Regressão", "Datasets"])

        with abas[0]:
            self._clustering()

        with abas[1]:
            self._classificacao()

        with abas[2]:
            self._regressao()

        with abas[3]:
            self._datasets()

        return "Métricas Calculadas"