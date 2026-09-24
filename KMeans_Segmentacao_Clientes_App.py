import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title='Segmentação Inteligente de Clientes', layout='wide')

st.title('Segmentação Inteligente de Clientes com K-Means')
st.caption('Descoberta de grupos de consumidores com comportamentos semelhantes para apoiar campanhas mais precisas.')

st.markdown('''
### Problema de negócio
**Como segmentar o público com mais precisão para campanhas eficazes?**

A segmentação tradicional normalmente parte de regras previamente definidas, como idade, localização ou faixa de renda.
Neste sistema, o **K-Means** procura padrões diretamente nos dados e agrupa clientes que apresentam comportamentos semelhantes,
mesmo quando esses consumidores parecem diferentes à primeira vista.

O objetivo não é apenas formar grupos: é transformar os clusters em **segmentos de marketing acionáveis**.
''')

with st.expander('Como funciona o processo de IA?', expanded=False):
    st.markdown('''
1. **Carregamento da base:** cada linha representa um cliente e cada coluna representa uma característica.
2. **Seleção das variáveis:** você escolhe quais variáveis numéricas deverão representar o comportamento do cliente.
3. **Padronização:** as variáveis são colocadas em uma escala comparável para evitar que valores monetários, por exemplo, dominem variáveis menores.
4. **Escolha de K:** o sistema testa diferentes quantidades de clusters e calcula o **coeficiente de silhueta**.
5. **Clustering:** o K-Means agrupa os clientes em torno de centróides.
6. **Perfil dos segmentos:** o sistema calcula as médias de cada variável por cluster.
7. **Ação de marketing:** os perfis podem ser usados para definir campanhas diferenciadas.
''')

st.divider()

uploaded_file = st.file_uploader('Carregue uma base de clientes em CSV', type=['csv'])

if uploaded_file is None:
    st.info('Carregue um arquivo CSV para iniciar a análise. O arquivo deve conter uma linha por cliente e, preferencialmente, várias colunas numéricas de comportamento.')
    st.markdown('''
**Exemplos de variáveis úteis:**
- valor gasto no período;
- frequência de compras;
- ticket médio;
- itens por compra;
- dias desde a última compra;
- percentual de compras com desconto;
- compras no aplicativo;
- compras na loja física;
- número de categorias compradas;
- taxa de resposta a campanhas anteriores.
''')
    st.stop()

try:
    df_original = pd.read_csv(uploaded_file)
except UnicodeDecodeError:
    uploaded_file.seek(0)
    df_original = pd.read_csv(uploaded_file, encoding='latin-1')
except Exception as e:
    st.error(f'Não foi possível ler o arquivo CSV: {e}')
    st.stop()

if df_original.empty:
    st.error('A base carregada está vazia.')
    st.stop()

st.subheader('1. Base carregada')
col1, col2, col3 = st.columns(3)
col1.metric('Clientes', f'{len(df_original):,}'.replace(',', '.'))
col2.metric('Variáveis', df_original.shape[1])
col3.metric('Valores ausentes', int(df_original.isna().sum().sum()))

st.dataframe(df_original.head(20), use_container_width=True)

numeric_cols = df_original.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error('A base precisa ter pelo menos duas colunas numéricas para executar o K-Means.')
    st.stop()

st.subheader('2. Escolha das variáveis para segmentação')

st.markdown('''
Selecione as características que representam o **comportamento do consumidor**. Evite usar códigos ou identificadores de cliente,
pois eles não representam comportamento e podem distorcer os clusters.
''')

selected_cols = st.multiselect(
    'Variáveis usadas pelo K-Means',
    options=numeric_cols,
    default=numeric_cols[:min(5, len(numeric_cols))]
)

if len(selected_cols) < 2:
    st.warning('Selecione pelo menos duas variáveis numéricas.')
    st.stop()

work_df = df_original[selected_cols].copy()

missing_before = int(work_df.isna().sum().sum())
if missing_before > 0:
    st.warning(f'Foram encontrados {missing_before} valores ausentes nas variáveis selecionadas. O sistema preencherá cada ausência com a mediana da respectiva variável.')
    for col in selected_cols:
        work_df[col] = work_df[col].fillna(work_df[col].median())

if work_df.nunique().min() <= 1:
    constant_cols = work_df.columns[work_df.nunique() <= 1].tolist()
    st.error('As seguintes variáveis não possuem variação e precisam ser removidas da análise: ' + ', '.join(constant_cols))
    st.stop()

st.markdown('**Por que padronizar?** O K-Means usa distâncias. Sem padronização, uma variável como gasto anual em reais pode ter muito mais peso que uma variável como frequência de compras apenas porque sua escala numérica é maior.')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(work_df)

st.subheader('3. Descoberta do número de segmentos')

max_k = min(10, len(work_df) - 1)
if max_k < 2:
    st.error('São necessários mais registros para formar clusters.')
    st.stop()

candidate_ks = list(range(2, max_k + 1))
silhouette_scores = []
inertias = []

for candidate_k in candidate_ks:
    model_test = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
    labels_test = model_test.fit_predict(X_scaled)
    inertias.append(model_test.inertia_)
    if len(set(labels_test)) > 1:
        silhouette_scores.append(silhouette_score(X_scaled, labels_test))
    else:
        silhouette_scores.append(np.nan)

valid_scores = [(k, s) for k, s in zip(candidate_ks, silhouette_scores) if not np.isnan(s)]
best_k = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else 2

left, right = st.columns(2)

with left:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(candidate_ks, silhouette_scores, marker='o')
    ax1.set_xlabel('Número de clusters (K)')
    ax1.set_ylabel('Coeficiente de silhueta')
    ax1.set_title('Qualidade da separação entre os clusters')
    ax1.grid(alpha=0.25)
    st.pyplot(fig1)
    plt.close(fig1)

with right:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(candidate_ks, inertias, marker='o')
    ax2.set_xlabel('Número de clusters (K)')
    ax2.set_ylabel('Inércia')
    ax2.set_title('Método do cotovelo')
    ax2.grid(alpha=0.25)
    st.pyplot(fig2)
    plt.close(fig2)

st.success(f'Pelo coeficiente de silhueta, o melhor resultado entre os valores testados foi **K = {best_k}**.')

st.markdown('''
**Interpretação:**
- **Silhueta:** quanto maior, melhor a combinação entre coesão interna do grupo e separação dos demais grupos.
- **Cotovelo:** procura um ponto a partir do qual aumentar K reduz pouco a inércia.
- O resultado matemático deve ser confrontado com a **utilidade gerencial**. Nem sempre o maior índice produz a segmentação mais útil para uma campanha.
''')

k = st.slider(
    'Escolha o número de segmentos que deseja usar',
    min_value=2,
    max_value=max_k,
    value=int(best_k)
)

st.subheader('4. Segmentação com K-Means')

model = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

result_df = df_original.copy()
result_df['Cluster'] = clusters
result_df['Segmento'] = result_df['Cluster'].apply(lambda x: f'Segmento {x + 1}')

sil = silhouette_score(X_scaled, clusters) if len(set(clusters)) > 1 else np.nan

m1, m2, m3 = st.columns(3)
m1.metric('Segmentos criados', k)
m2.metric('Silhueta do modelo', f'{sil:.3f}' if not np.isnan(sil) else 'N/A')
m3.metric('Clientes segmentados', len(result_df))

st.subheader('5. Visualização dos grupos')

if len(selected_cols) == 2:
    plot_x = work_df[selected_cols[0]].values
    plot_y = work_df[selected_cols[1]].values
    x_label = selected_cols[0]
    y_label = selected_cols[1]
    title = 'Clusters usando as duas variáveis selecionadas'
else:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    plot_x = coords[:, 0]
    plot_y = coords[:, 1]
    x_label = 'Componente principal 1'
    y_label = 'Componente principal 2'
    explained = pca.explained_variance_ratio_.sum() * 100
    title = f'Visão 2D dos clusters via PCA ({explained:.1f}% da variância representada)'

fig3, ax3 = plt.subplots(figsize=(10, 6))
scatter = ax3.scatter(plot_x, plot_y, c=clusters, s=65, alpha=0.75, cmap='tab10')
ax3.set_xlabel(x_label)
ax3.set_ylabel(y_label)
ax3.set_title(title)
ax3.grid(alpha=0.2)
legend = ax3.legend(*scatter.legend_elements(), title='Cluster')
ax3.add_artist(legend)
st.pyplot(fig3)
plt.close(fig3)

if len(selected_cols) > 2:
    st.caption('O PCA é usado apenas para mostrar visualmente uma base multidimensional em duas dimensões. O K-Means continua sendo executado com todas as variáveis selecionadas.')

st.subheader('6. Perfil dos segmentos')

profile = result_df.groupby('Segmento')[selected_cols].mean().round(2)
counts = result_df.groupby('Segmento').size().rename('Clientes')
profile_with_count = profile.copy()
profile_with_count.insert(0, 'Clientes', counts)
profile_with_count['% da base'] = (counts / len(result_df) * 100).round(1)

st.dataframe(profile_with_count, use_container_width=True)

st.markdown('### Leitura automática dos segmentos')

overall_means = work_df.mean()
overall_stds = work_df.std(ddof=0).replace(0, np.nan)
cluster_means = result_df.groupby('Segmento')[selected_cols].mean()

for segment in cluster_means.index:
    z = ((cluster_means.loc[segment] - overall_means) / overall_stds).sort_values()
    low = z.head(min(2, len(z))).index.tolist()
    high = z.tail(min(2, len(z))).index.tolist()[::-1]
    count = int(counts.loc[segment])
    pct = count / len(result_df) * 100

    st.markdown(f'**{segment} — {count} clientes ({pct:.1f}%)**')
    st.write(
        'Características relativamente mais altas: ' + ', '.join(high) + '. '
        'Características relativamente mais baixas: ' + ', '.join(low) + '.'
    )

st.info('Os nomes dos segmentos acima são neutros de propósito. A equipe de marketing deve interpretar o perfil e atribuir rótulos de negócio, como “clientes de alto valor”, “ocasionais sensíveis a desconto” ou “digitais recorrentes”, quando os dados realmente sustentarem essa interpretação.')

st.subheader('7. Da segmentação para a campanha')

st.markdown('''
Após identificar os clusters, a empresa pode criar uma estratégia diferente para cada grupo. Exemplos:

- **alto gasto + alta frequência:** campanhas de fidelização, benefícios exclusivos e acesso antecipado;
- **alto gasto + baixa frequência:** ações de recompra e relacionamento;
- **baixo gasto + alta frequência:** bundles, cross-sell e aumento do ticket médio;
- **muita sensibilidade a desconto:** campanhas promocionais direcionadas;
- **alto engajamento digital:** campanhas personalizadas em canais digitais;
- **clientes inativos:** campanhas de reativação.

O ponto central é que os segmentos são **descobertos pelos padrões dos dados**, e não definidos antecipadamente apenas por critérios demográficos.
''')

st.subheader('8. Base final para uso em campanhas')
st.dataframe(result_df.head(50), use_container_width=True)

csv_bytes = result_df.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    'Baixar base segmentada em CSV',
    data=csv_bytes,
    file_name='clientes_segmentados_kmeans.csv',
    mime='text/csv'
)

st.markdown('''
---
### Conceitos-chave
**K-Means:** algoritmo de aprendizado não supervisionado que procura formar K grupos, minimizando a distância dos clientes em relação ao centróide do próprio grupo.

**Cluster:** grupo de registros semelhantes segundo as variáveis utilizadas pelo algoritmo.

**Centróide:** ponto médio que representa matematicamente um cluster.

**Aprendizado não supervisionado:** o sistema não recebe previamente uma coluna dizendo qual é o segmento correto. Ele procura padrões estruturais nos próprios dados.

**Personalização em escala:** depois de segmentar milhares ou milhões de clientes, a empresa pode executar estratégias diferenciadas para cada grupo de forma sistemática.
''')
