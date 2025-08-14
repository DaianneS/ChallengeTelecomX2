# ChallengeTelecomX2

# 📊 Análise Preditiva de Evasão de Clientes - TelecomX

Este repositório contém uma análise completa e a construção de modelos de Machine Learning para prever a evasão de clientes (churn) na empresa fictícia de telecomunicações TelecomX.

## 🚀 Tecnologias Utilizadas
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-88d498?style=for-the-badge&logo=seaborn&logoColor=white)

- **Python 3**
- **Pandas:** Para manipulação e análise de dados.
- **Scikit-learn:** Para pré-processamento, modelagem e avaliação.
- **Imbalanced-learn:** Para balanceamento de classes com SMOTE.
- **Seaborn & Matplotlib:** Para visualização de dados.
- **Jupyter Notebook:** Como ambiente de desenvolvimento.


## 🎯 Visão Geral

A evasão de clientes, ou *Churn*, é uma das maiores ameaças para empresas de telecomunicações. Um alto índice de Churn não apenas afeta a receita, mas também aumenta os custos de aquisição e pode prejudicar a imagem da marca. Este projeto utiliza dados históricos de clientes para construir modelos preditivos capazes de identificar os principais fatores de risco e antecipar quais clientes têm maior probabilidade de cancelar seus serviços, permitindo que a TelecomX atue de forma proativa com estratégias de retenção.

## 🛠️ Metodologia

O projeto foi estruturado em um pipeline completo de Data Science, desde a preparação dos dados até a avaliação e interpretação dos modelos.

### 1. Pré-processamento e Análise Exploratória de Dados (EDA)

- **Limpeza de Dados:** Colunas irrelevantes para a predição, como `ID_Cliente`, foram removidas. A variável alvo `Cancelou` foi transformada de categórica ("Sim"/"Não") para numérica (1/0).
- **Codificação de Variáveis Categóricas:** A técnica de *One-Hot Encoding* foi aplicada a todas as colunas categóricas (ex: `Tipo_Contrato`, `Metodo_Pagamento`) para convertê-las em um formato numérico que os modelos pudessem processar.
- **Análise de Correlação:** Foi gerada uma matriz de correlação para identificar as variáveis com maior impacto na evasão de clientes. As mais correlacionadas positivamente com o churn foram:
    - `Tipo_Contrato_Mês a mês`
    - `Tipo_Internet_Fibra ótica`
    - `Metodo_Pagamento_Cheque eletrônico`
- **Visualizações:** Gráficos foram gerados para entender a relação entre a evasão e fatores como o tipo de contrato, a cobrança total e os meses de permanência.

### 2. Preparação dos Dados para Modelagem

- **Balanceamento de Classes:** A análise inicial revelou um conjunto de dados desbalanceado (73.46% de não evasão vs. 26.54% de evasão). Para corrigir isso e evitar um modelo enviesado, foi aplicada a técnica de oversampling **SMOTE** apenas nos dados de treino, criando um conjunto de dados balanceado para a modelagem.
- **Padronização de Dados:** Para modelos sensíveis à escala das features (como a Regressão Logística), as variáveis numéricas (`Meses_Permanencia`, `Cobranca_Mensal`, `Cobranca_Total`) foram padronizadas utilizando o `StandardScaler`.
- **Divisão em Treino e Teste:** O conjunto de dados foi dividido em 75% para treinamento e 25% para teste, utilizando a estratificação para manter a proporção original de churn em ambos os conjuntos.

### 3. Modelagem Preditiva

Foram treinados e avaliados dois modelos distintos para comparação:

1.  **Regressão Logística:** Um modelo linear, simples e de fácil interpretação, que serve como uma excelente linha de base.
2.  **Random Forest:** Um modelo de conjunto (*ensemble*) baseado em árvores de decisão, que não é sensível à escala dos dados e é capaz de capturar relações não-lineares complexas.

## 📈 Resultados dos Modelos

A avaliação foi focada em métricas relevantes para o problema de negócio, como **Acurácia**, **Precisão** e, principalmente, **Recall** (a capacidade do modelo de identificar corretamente os clientes que irão evadir).

| Métrica | Regressão Logística | Random Forest |
| :--- | :---: | :---: |
| Acurácia (Teste) | 0.7428 | **0.7785** |
| Precisão (Teste) | 0.5097 | **0.5828** |
| Recall (Teste) | **0.7880** | 0.5803 |
| F1-Score (Teste) | 0.6190 | **0.5815** |


O modelo de **Random Forest** demonstrou um desempenho geral mais equilibrado e uma acurácia superior, sendo o modelo recomendado para a implementação.

## 🔑 Principais Fatores de Evasão (Feature Importance)

A análise de importância de variáveis do modelo Random Forest destacou os seguintes fatores como os mais preditivos para o churn:

1.  **Meses de Permanência:** A variável mais importante, indicando que a maior parte da evasão ocorre nos primeiros meses de contrato.
2.  **Cobrança Total:** Clientes com uma cobrança total acumulada menor (reflexo de pouco tempo de contrato) tendem a evadir mais.
3.  **Cobrança Mensal:** Faturas mensais mais altas representam um fator de risco.
4.  **Tipo de Contrato (Mês a Mês):** A ausência de um contrato de longo prazo é um dos indicadores mais fortes de uma futura evasão.

## 💡 Recomendações Estratégicas

Com base nos resultados, as seguintes ações são recomendadas para a TelecomX:

- **Incentivar Contratos de Longo Prazo:** Criar campanhas de migração agressivas, oferecendo descontos ou benefícios para clientes com contrato mensal que optarem por planos anuais ou bianuais.
- **Programa de Onboarding Intensivo:** Focar os esforços de retenção nos clientes novos, especialmente nos primeiros seis meses, para garantir que eles percebam o valor do serviço.
- **Monitorar Cobranças Elevadas:** Implementar um sistema de alerta para clientes com faturas altas, oferecendo revisões de plano ou pacotes de fidelidade.
- **Otimizar Processos de Pagamento:** Investigar a causa do alto churn entre usuários de cheque eletrônico e incentivar a migração para métodos de pagamento automáticos.

## ⚙️ Como Utilizar este Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
    cd nome-do-repositorio
    ```
2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Se um arquivo `requirements.txt` não estiver disponível, instale as bibliotecas abaixo:*
    ```bash
    pip install pandas scikit-learn imbalanced-learn seaborn matplotlib jupyter
    ```
4.  **Execute o Notebook:**
    - Certifique-se de que o arquivo de dados `TelecomX_data_tratados.csv` esteja no mesmo diretório.
    - Inicie o Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    - Abra o arquivo `telecomx_parte2_BR.ipynb` e execute as células.
