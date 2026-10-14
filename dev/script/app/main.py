import sys
from pathlib import Path

# main.py -> app -> script -> dev -> raiz do repo
RAIZ = Path(__file__).resolve().parents[3]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from dev.script.src.acesso_data import acesso_data_logo, acesso_data_icon
from dev.script.src.paginas import Paginas

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from dev.script.src.acesso_data import acesso_data_logo, acesso_data_icon
from dev.script.src.paginas import Paginas


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
    # Deve ser o primeiro comando do Streamlit no script
    st.set_page_config(
        page_title="Variantes Neuro ML",
        layout="wide",
        page_icon=acesso_data_icon(),
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

    title = "Métricas dos Modelos de Classificação - Variantes em Doença Neurológica"
    st.markdown(
        f"<br><h3 style='text-align: center; '>{title}</h3>", unsafe_allow_html=True
    )

    return "Ajustado"


@st.cache_data
def logo_projeto() -> str:
    """
    Função de inserção do logo do projeto (opcional).
    Parametros:
        None
    Retorna:
        str: Mensagem de sucesso.
    """
    caminho = acesso_data_logo()
    if caminho is None:
        return "Sem logo"

    # Plota a imagem
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(np.asarray(Image.open(caminho)))
    plt.axis("off")

    # Edita as colunas e insere os dados
    col_img = st.columns((15, 1))
    with col_img[1]:
        st.pyplot(fig=fig)
    plt.close(fig)

    return "Logo Inserido"


if __name__ == "__main__":
    pagina_estrutura()
    pagina = Paginas()
    pagina.metricas()
    logo_projeto()