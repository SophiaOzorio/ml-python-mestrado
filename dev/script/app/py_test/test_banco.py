import os
from dev.script.app.src.paginas import _busca_splits


def test_busca_dados_cluster():
    # Given / When: _busca_splits retorna (train, valid, test)
    df_train, df_valid, df_test = _busca_splits()
    
    # Then
    assert len([df_train, df_valid, df_test]) == 3


def test_busca_dados_class():
    # Given / When
    df_train, _, df_test = _busca_splits()

    # Then
    assert len(df_train) > 0
    assert len(df_test) > 0


def test_busca_dados_regr():
    # Given / When
    df_train, _, df_test = _busca_splits()

    # Then
    assert len(df_train) > 0
    assert len(df_test) > 0