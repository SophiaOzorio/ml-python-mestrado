import sys
from pathlib import Path

# main.py -> app -> script -> dev -> raiz do repo
RAIZ = Path(__file__).resolve().parents[3]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

from dev.script.src.paginas import Paginas  # noqa: E402


def pagina_estrutura() -> str:
    """
    Função de estruturação da página inicial.
    Retorna:
        str: Mensagem de sucesso.
    """
    # Deve ser o primeiro comando do Streamlit no script
    st.set_page_config(page_title="Variantes Neuro ML", layout="wide")
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

    title = "Métricas dos Modelos - Variantes em Doença Neurológica"
    st.markdown(
        f"<br><h3 style='text-align: center; '>{title}</h3>", unsafe_allow_html=True
    )

    return "Ajustado"


if __name__ == "__main__":
    pagina_estrutura()
    Paginas().metricas()