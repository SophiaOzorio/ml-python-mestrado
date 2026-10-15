from main import pagina_estrutura


def test_estrutura_pagina():
    # Given
    teste_entrada = "Ajustado"

    # When
    teste_saida = pagina_estrutura()

    # Then
    assert teste_entrada == teste_saida
