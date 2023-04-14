# Tema: 'Utilização de Modelos Baseados em Árvores de Decisão para Elaboração de Defesas de Processos Trabalhistas'

# Descrição Resumida do Projeto
> Quando uma empresa recebe a notificação de um processo trabalhista, o departamento jurídico é acionado para representar o caso. Um dos maiores impasses durante a elaboração de uma defesa processual é entender qual a probabilidade de vitória para decidir o rumo que essa defesa irá seguir: tentar um acordo e encerrar o processo mais rápido e com menos custo (quando as chances de vitória do caso são baixas) ou seguir em frente com a ação visando a vitória, pois embora seja uma opção mais demorada, pode ter um menor custo quando a probabilidade de um resultado favorável for maior. 
>
> Desta forma, o objetivo do presente trabalho é auxiliar durante a elaboração dessas defesas. Então, o modelo aqui abordado foi projetado para prever o resultado desses processos trabalhistas, bem como a probabilidade do resultado e quais provas aumentam esse valor. Ao analisar a situação, percebe-se que se trata de um problema de classificação com duas classes (0: desfavorável para a empresa e 1: favorável) e para resolver foram utilizados alguns métodos baseados em árvores de decisão (Random Forest, LightGBM e EBM) para treinar o modelo e, posteriormente, comparou-se  o custo computacional e o desempenho, com base nas métricas de recall e precisão, a fim de escolher o mais eficaz. O método que teve o melhor desempenho dentre os estudados foi o LightGBM com acurácia de 0,782, precisão da classe 0 de 0,629, precisão da classe 1 de 0,845, recall da classe 0 de 0,629 e recall da classe 1 de 0,845.

# Metodologia
> O problema em questão estudado seguiu um fluxograma baseado na metodologia de desenvolvimento do Crisp-DM. Desse modo, foi dividido em quatro etapas: 
compreensão de dados e negócio; preparação e modelagem de dados; validação de lote; e implantação em lote.

# Ferramentas
> O projeto foi realizado em Python e utilizando o Azure Databricks.
>
> |Ferramenta/Biblioteca | Descrição|
> |--|--|
> |[Pandas](https://pandas.pydata.org/) |Biblioteca para manipulação e análise de dados| 
> |[Numpy](https://numpy.org/) |Biblioteca para cálculos matemáticos e estatísticos| 
> |[Scikit-learn](https://scikit-learn.org/stable/) |Biblioteca para modelagem preditiva e aprendizado de máquina| 
> |[Matplotlib](https://matplotlib.org/) e [Seaborn](https://seaborn.pydata.org/) |Bibliotecas para visualização de dados| 
> |[Azure Databricks](https://learn.microsoft.com/pt-br/azure/databricks/introduction/) |Plataforma de ferramentas para criar, implantar, compartilhar e manter soluções de dados de nível empresarial em escala| 

