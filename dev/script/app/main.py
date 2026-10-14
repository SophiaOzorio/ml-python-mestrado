import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from src.acesso_data import acesso_data_test, acesso_data_icon
from src.paginas import Paginas


def pagina_inicio(entrada: str) -> str:
    """
    Função de inicialização da página inicial.
    Parametros:
        entrada: str
    Retorna:
        str: Mensagem de sucesso.
    """
    return entrada


def pagina_estrutura() -> str:
    """
    Função de estruturação da página inicial.
    Parametros:
        None
    Retorna:
        str: Mensagem de sucesso.
    """
    st.set_page_config(
        page_title="FAcPyML", layout="wide", page_icon=acesso_data_icon()
    )
    st.markdown(
        """
        <style>
            .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = "Métricas de Modelos de Machine Learning"
    st.markdown(
        f"<br><h3 style='text-align: center; '>{title}</h3>", unsafe_allow_html=True
    )

    return "Ajustado"


@st.cache_data
def logo_datalab() -> str:
    """
    Função de inserção do logo do DataLab.
    Parametros:
        None
    Retorna:
        str: Mensagem de sucesso.
    """
    # Plota a imagem
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(np.asarray(Image.open(acesso_data_test())))
    plt.axis("off")

    # Edita as colunas e insere os dados
    col_img = st.columns((15, 1))
    with col_img[1]:
        st.pyplot(fig=fig)

    return "Logo Inserido"


if __name__ == "__main__":
    pagina_estrutura()
    pagina = Paginas()
    pagina.metricas()
    logo_datalab()
