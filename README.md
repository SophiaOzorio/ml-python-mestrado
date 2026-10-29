# Relatório Python para Machine Learning 

**Aluna:** Sophia Pereira Ozório — Mestrado  
**Disciplina:** Python para Machine Learning — 2026

## Sobre o projeto

Este projeto foi desenvolvido como parte da disciplina de Python para Machine Learning e contempla etapas de pré-processamento de dados, análise exploratória, classificação, clusterização, regressão e testagem.

O dataset utilizado foi gerado artificialmente com o auxílio do Claude e contém dados referentes à anotação de variantes genéticas identificadas em análises de sequenciamento do exoma completo de pacientes com suspeita de ataxia.

## Aplicação web

A aplicação desenvolvida com Streamlit reúne as funcionalidades do projeto e permite explorar as etapas de análise e modelagem dos dados.

**[Acessar a aplicação Streamlit](https://ml-python-mestrado-c36hjhgxd2ze6vycdvgg8s.streamlit.app/)**

## Estrutura do projeto

Os diretórios e arquivos estão organizados da seguinte forma, para facilitar a localização dos materiais e resultados.

### Notebooks

Diretório: `dev/script/`

1. `ydata-profiling-before.ipynb`
2. `limpeza_variantes.ipynb` 
3. `ydata-profiling_before.ipynb`
4. `split_transform.ipynb` 
5. `ydata-profiling-split.ipynb`
6. `treinamento_classif_variantes.ipynb`
7. `treinamento_cluster_variantes.ipynb`
8. `treinamento_regressao_variantes.ipynb`

### Modelos

Diretório: `dev/script/src/modelos/`

- `classifier.py` 
- `clustering.py`
- `regression.py`

### Dados

Diretório: `dev/script/data/`

Contém o dataset original utilizado no projeto.

### Resultados

Diretório: `dev/script/results/`

Contém os arquivos gerados durante as etapas de pré-processamento, análise e modelagem.

- **`split_dataset/`** — resultados da divisão do dataset em conjuntos de treino, validação e teste.
- **`reports/`** — relatórios gerados com o ydata-profiling.
- **Classificação** — DataFrames referentes aos modelos de classificação.
- **Clusterização** — DataFrames referentes aos modelos de clusterização.

### Testes

Diretório: `dev/script/app/py_test/`

Contém os testes implementados para verificar o funcionamento do projeto.
Os relatórios de cobertura dos testes estão disponíveis em `htmlcov/`.

## Apresentação

A apresentação realizada em **27/10/2026**, como parte da disciplina, está disponível no link abaixo.

**[Acessar a apresentação no Canva](https://canva.link/nh4dbjadqejiw22)**
