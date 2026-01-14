# TELECOM MACHINE LEARNING
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- **Descrição do problema**: Predizer evasão de clientes (churn) em uma empresa de telecomunicações, isto é, identificar se um cliente irá cancelar seu serviço (Churn = Yes/No). O dataset foi disponibilizado pela IBM e simula uma operadora fixa/internet com 7.043 clientes fictícios de uma empresa de telefonia na Califórnia. Cada registro inclui variáveis sobre os serviços do cliente (se tem internet, plano de telefone, TV a cabo, etc.), informações de conta (tipo de contrato mensal ou anual, cobrança eletrônica, valores mensais e totais) e dados demográficos básicos (sexo, se é idoso, se tem dependentes). O atributo alvo “Churn” indica se o cliente deixou o serviço no último mês. Trata-se de um caso realista de classificação binária desbalanceada (cerca de 26% dos clientes no conjunto são churn, e 74% não churn. Modelos supervisonados como regressão logística e Random Forest já demonstram bom desempenho (~80–85% acurácia) nessa tarefa, e a análise dos features pode revelar insights, como identificar que tipo de contrato ou serviço está associado a maior risco de churn.
    - https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113#:~:text=The%20Telco%20customer%20churn%20data,CLTV%29%20index
    - https://medium.com/@UjuEmmanuella/data-driven-customer-retention-the-churn-analysis-project-that-saved-4-3m-a0078fa8518d#:~:text=Data,gauge%20visualization%20immediately%20communicates
    - https://www.turintech.ai/cases/customer-churn-prediction-using-machine-learning-a-step-by-step-guide-with-evoml#:~:text=For%20this%20analysis%2C%20we%20consider,As%20such%2C%20Churn%20is%20our

- **Fonte dos dados**: Disponível publicamente, por exemplo no Kaggle: Telco Customer Churn dataset (IBM). Esse conjunto de dados foi originalmente distribuído como parte de exemplos da plataforma IBM Cognos Analytics. No Kaggle, os dados vêm geralmente em um arquivo CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv). Não há texto ou imagens complexas – são dados estruturados (colunas numéricas/boolianas/categóricas simples), apropriados para manipulação com pandas.
    - https://www.turintech.ai/cases/customer-churn-prediction-using-machine-learning-a-step-by-step-guide-with-evoml#:~:text=For%20this%20analysis%2C%20we%20consider,As%20such%2C%20Churn%20is%20our


- **Estudos prévios**: A problemática de churn em telecom é bastante estudada na literatura de Data Science. Especificamente, este dataset IBM tem sido explorado em diversos artigos de blog e repositórios. Por exemplo, a empresa TurinTech publicou um case study mostrando passo a passo a construção de um modelo de churn com este conjunto de dados. Nesse artigo, eles confirmam que o conjunto contém 7043 clientes e 21 variáveis relevantes, e utilizaram algoritmos de Machine Learning interpretáveis para destacar fatores de churn (descobrindo, por exemplo, que clientes com contratos mensais têm probabilidade de cancelamento bem maior que aqueles com contratos anuais)

    Na literatura acadêmica, problemas similares de churn foram abordados com técnicas de árvores de decisão, florestas aleatórias e boosting, frequentemente buscando identificar quais atributos (como satisfação, tipo de contrato, uso de serviços) mais contribuem para a saída do cliente. Para inovar em cima do que já foi feito, você poderia focar em análise de explicabilidade (por exemplo, usando SHAP para entender o impacto de cada atributo) ou em estratégias de amostragem/ponderação para tratar o desbalanceamento, apresentando recomendações de retenção de clientes baseadas nos insights do modelo.

    - https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113#:~:text=Scenario%204

    - https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113#:~:text=In%20the%20Telco%20churn%20dashboard%2C,in%20the%20San%20Diego%20area

    - https://www.turintech.ai/cases/customer-churn-prediction-using-machine-learning-a-step-by-step-guide-with-evoml#:~:text=In%20this%20dataset%2C%20Contract%20is,percentage%20of%20customers%20who%20churn

    - https://www.turintech.ai/cases/customer-churn-prediction-using-machine-learning-a-step-by-step-guide-with-evoml#:~:text=For%20this%20analysis%2C%20we%20consider,As%20such%2C%20Churn%20is%20our