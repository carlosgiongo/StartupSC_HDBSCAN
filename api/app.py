from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px
import plotly.graph_objects as go
import os

app = FastAPI(
    title="API de Visualização de Clusters HDBSCAN",
    description="Esta API executa um pipeline de clusterização e retorna uma visualização 3D interativa.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

model = SentenceTransformer('all-mpnet-base-v2')

COLUMNS_TO_VECTORIZE = ['Negócio', 'Pitch']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DF_LOAD_PATH = os.path.join(DATA_DIR, "dataframe_with_embeddings.pkl")
EMBEDDINGS_LOAD_PATH = os.path.join(DATA_DIR, "embeddings_array.npy")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def executar_pipeline_e_gerar_grafico(nome:str, pitch: str, descricao: str):
    try:
        print("Iniciando pipeline: Carregando dados...")
        if not os.path.exists(DF_LOAD_PATH) or not os.path.exists(EMBEDDINGS_LOAD_PATH):
            error_msg = f"Arquivos de dados não encontrados. Verifique os caminhos: {DF_LOAD_PATH}, {EMBEDDINGS_LOAD_PATH}"
            raise FileNotFoundError(error_msg)

        df_filtered = pd.read_pickle(DF_LOAD_PATH)
        embeddings_array = np.load(EMBEDDINGS_LOAD_PATH)

        ################################ ATRIBUIÇÃO DE NOVOS DADOS ##################################################################
        if pitch and descricao:
            processed_pitch = pitch.lstrip()
            processed_descricao = descricao.lstrip()
            processed_nome = nome.lstrip() if nome else 'Nova Startup'
            texts_to_join = [text for text in [processed_descricao, processed_pitch] if text]
            combined_text = ' + '.join(texts_to_join)

            print("Gerando embedding para os novos dados...")
            
            new_embedding = model.encode([combined_text])[0]
            new_data = {
                'Startup': processed_nome,
                'Negócio': processed_pitch,
                'Pitch': processed_descricao,
                'Ano': pd.Timestamp.now().year,
                'texto_combinado_para_vetorizar': combined_text,
                'vetor_embedding': new_embedding
            }

            print("Adicionando nova linha ao DataFrame...")
            new_row_df = pd.DataFrame([new_data]) 
            df_filtered = pd.concat([df_filtered, new_row_df], ignore_index=True)
            embeddings_array = np.vstack((embeddings_array, new_embedding))

            print("Novos dados atribuídos com sucesso!")
        
        ##############################################################################################################################
        
        print("Dados carregados. Validando...")
        if 'df_filtered' in locals() and 'embeddings_array' in locals():
            if len(df_filtered) == embeddings_array.shape[0]:
                print(f"Validação OK: DataFrame ({len(df_filtered)}) e Embeddings ({embeddings_array.shape[0]}) correspondem.")
            else:
                msg = f"INCONSISTÊNCIA: DataFrame ({len(df_filtered)}) e Embeddings ({embeddings_array.shape[0]}) NÃO correspondem."
                print(msg)
                raise ValueError(msg)
        else:
            msg = "Erro ao carregar df_filtered ou embeddings_array."
            print(msg)
            raise ValueError(msg)
        
        CLUSTER_LABEL_COL = 'hdbscan_cluster'

        print("Iniciando clusterização HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean',
            algorithm='best',
            gen_min_span_tree=True, 
            cluster_selection_method='leaf',
        )
        clusters = clusterer.fit_predict(embeddings_array)
        df_filtered[CLUSTER_LABEL_COL] = clusters
        print(f"Clusterização HDBSCAN inicial concluída. Clusters: {np.unique(clusters)}")

        print("Aplicando tratamento de ruído com threshold...")
        NOISE_ASSIGNMENT_THRESHOLD = 0.77 
        hdbscan_labels_from_fit = clusters
        final_labels = hdbscan_labels_from_fit.copy()

        original_noise_indices = np.where(hdbscan_labels_from_fit == -1)[0]
        original_clustered_indices = np.where(hdbscan_labels_from_fit != -1)[0]

        num_reassigned_within_threshold = 0
        
        if len(original_noise_indices) > 0:
            if len(original_clustered_indices) > 0:
                print(f"Encontrados {len(original_noise_indices)} pontos de ruído. Tentando reatribuir...")
                actual_cluster_ids = np.unique(hdbscan_labels_from_fit[original_clustered_indices])
                cluster_centroids = {}

                for cluster_id_iter in actual_cluster_ids:
                    points_in_cluster_mask = (hdbscan_labels_from_fit == cluster_id_iter)
                    embeddings_in_cluster = embeddings_array[points_in_cluster_mask]
                    if embeddings_in_cluster.shape[0] > 0:
                        cluster_centroids[cluster_id_iter] = np.mean(embeddings_in_cluster, axis=0)

                if cluster_centroids:
                    for noise_idx in original_noise_indices:
                        noise_embedding = embeddings_array[noise_idx]
                        min_dist = float('inf')
                        assigned_cluster_id = -1

                        for cluster_id_iter, centroid in cluster_centroids.items():
                            dist = np.linalg.norm(noise_embedding - centroid)
                            if dist < min_dist:
                                min_dist = dist
                                assigned_cluster_id = cluster_id_iter

                        if assigned_cluster_id != -1 and min_dist <= NOISE_ASSIGNMENT_THRESHOLD:
                            final_labels[noise_idx] = assigned_cluster_id
                            num_reassigned_within_threshold +=1
                        else:
                            final_labels[noise_idx] = -1
                print(f"{num_reassigned_within_threshold} pontos de ruído reatribuídos dentro do threshold.")
            else:
                print("Todos os pontos foram classificados como ruído inicialmente. Nenhum cluster para reatribuir.")
        else:
            print("Nenhum ponto de ruído encontrado na clusterização inicial.")

        # next_cluster_id = 0
        # if len(existing_cluster_ids_in_final) > 0:
        #     next_cluster_id = -1
        #
        
        true_outlier_indices = np.where(final_labels == -1)[0]
        
        if len(true_outlier_indices) > 0:
            existing_positive_cluster_ids = np.unique(final_labels[final_labels >= 0])
            next_new_cluster_id = 0
            if len(existing_positive_cluster_ids) > 0:
                next_new_cluster_id = np.max(existing_positive_cluster_ids) + 1
            
            # for outlier_idx in true_outlier_indices:
            #     final_labels[outlier_idx] = next_new_cluster_id
            #     next_new_cluster_id += 1

        df_filtered[CLUSTER_LABEL_COL] = final_labels
        print(f"Tratamento de ruído finalizado. Clusters finais: {np.unique(final_labels)}")

        print("Iniciando redução de dimensionalidade UMAP...")
        N_NEIGHBORS = 10
        MIN_DIST = 0.02
        N_COMPONENTS = 3
        METRIC = 'euclidean'
        RANDOM_STATE = 42

        reducer = umap.UMAP(
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            n_components=N_COMPONENTS,
            metric=METRIC,
            random_state=RANDOM_STATE
        )
        embeddings_3d = reducer.fit_transform(embeddings_array)
        df_filtered['umap_x'] = embeddings_3d[:, 0]
        df_filtered['umap_y'] = embeddings_3d[:, 1]
        df_filtered['umap_z'] = embeddings_3d[:, 2]
        print("Redução UMAP concluída.")

        print("Preparando dados para Plotly...")
        if CLUSTER_LABEL_COL in df_filtered.columns:
            df_filtered['hdbscan_cluster_str'] = df_filtered[CLUSTER_LABEL_COL].astype(str)
            df_filtered.loc[df_filtered[CLUSTER_LABEL_COL] == -1, 'hdbscan_cluster_str'] = 'Noise (-1)'
        else:
            msg = f"ERRO: Coluna de cluster '{CLUSTER_LABEL_COL}' não encontrada para plotagem."
            print(msg)
            raise ValueError(msg)

        max_hover_text_len = 150
        if 'Startup' in df_filtered.columns:
            df_filtered['hover_text_for_plot'] = df_filtered['Startup'].astype(str).str[:max_hover_text_len] + '...'
        else:
            print("AVISO: Coluna 'Startup' não encontrada. Usando índice para hover_text.")
            df_filtered['hover_text_for_plot'] = "Índice: " + df_filtered.index.astype(str)

        hover_data_columns = [CLUSTER_LABEL_COL, 'hover_text_for_plot']
        
        # if isinstance(CLUSTER_LABEL_COL, str):
        #     hover_data_columns = [CLUSTER_LABEL_COL, 'hover_text_for_plot']
        # elif isinstance(CLUSTER_LABEL_COL, list):
        #     hover_data_columns = CLUSTER_LABEL_COL + ['hover_text_for_plot']

        color_map = {"Noise (-1)": "lightgrey"}

        if 'Ano' not in df_filtered.columns:
            msg = "ERRO: Coluna 'Ano' não encontrada no DataFrame para o filtro do gráfico."
            print(msg)
            raise ValueError(msg)
        
        df_filtered['Ano'] = df_filtered['Ano'].astype(str)
        unique_years = sorted(df_filtered['Ano'].unique())
        fig = go.Figure()

        df_all = df_filtered.copy()
        if not df_all.empty:
            temp_fig_all = px.scatter_3d(
                df_all, x='umap_x', y='umap_y', z='umap_z',
                color='hdbscan_cluster_str', hover_data=hover_data_columns,
                color_discrete_map=color_map, opacity=0.8,
                labels={'hdbscan_cluster_str': 'Cluster HDBSCAN'}
            )
            for t in temp_fig_all.data:
                t.visible = True
                fig.add_trace(t)
        num_traces_all = len(fig.data)

        year_trace_info = {}
        for year_val in unique_years:
            df_year = df_filtered[df_filtered['Ano'] == year_val].copy()
            start_idx_for_this_year = len(fig.data)
            num_traces_for_this_year = 0
            if not df_year.empty:
                temp_fig_year = px.scatter_3d(
                    df_year, x='umap_x', y='umap_y', z='umap_z',
                    color='hdbscan_cluster_str', hover_data=hover_data_columns,
                    color_discrete_map=color_map, opacity=0.8,
                    labels={'hdbscan_cluster_str': 'Cluster HDBSCAN'}
                )
                for t in temp_fig_year.data:
                    t.visible = False
                    t.showlegend = False
                    fig.add_trace(t)
                    num_traces_for_this_year += 1
            year_trace_info[year_val] = (start_idx_for_this_year, num_traces_for_this_year)
        
        buttons = []
        visibility_for_all_button = [True] * num_traces_all + [False] * (len(fig.data) - num_traces_all)
        buttons.append(dict(label='Todos os Anos', method='update',
                            args=[{'visible': visibility_for_all_button},
                                  {'title.text': 'Visualização 3D dos Clusters (Todos os Anos)'}]))
        for year_val in unique_years:
            visibility_for_year_button = [False] * len(fig.data)
            start_idx, num_traces_year = year_trace_info.get(year_val, (0,0))
            if num_traces_year > 0:
                for i in range(start_idx, start_idx + num_traces_year):
                    visibility_for_year_button[i] = True
            buttons.append(dict(label=f'Ano: {year_val}', method='update',
                                args=[{'visible': visibility_for_year_button},
                                      {'title.text': f'Visualização 3D dos Clusters (Ano: {year_val})'}]))
        
        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, direction="down",
                              pad={"r": 10, "t": 10}, showactive=True,
                              x=0.01, xanchor="left", y=1.12, yanchor="top")],
            title_text="Visualização 3D dos Clusters HDBSCAN (Redução UMAP)", title_x=0.5,
            scene=dict(xaxis_title='UMAP X', yaxis_title='UMAP Y', zaxis_title='UMAP Z'),
            margin=dict(l=0, r=0, b=0, t=50), legend_title_text='Cluster HDBSCAN'
        )
        fig.update_traces(marker=dict(size=3))
        html_output = fig.to_html(full_html=True, include_plotlyjs='cdn')
        return html_output

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: Arquivo de dados não encontrado. {e}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: Problema com os dados. {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor ao gerar visualização: {e}")

class ClusterInput(BaseModel):
    nome: str
    pitch: str
    descricao: str

@app.post("/visualizacao-clusters", response_class=HTMLResponse)
async def post_cluster_visualization(dados: ClusterInput):
    html_content = executar_pipeline_e_gerar_grafico(dados.nome, dados.pitch, dados.descricao)
    
    if html_content:
        print("Retornando HTML da visualização.")
        return HTMLResponse(content=html_content)
    else:
        print("Falha ao gerar HTML, retornando erro 500.")
        raise HTTPException(status_code=500, detail="Falha ao gerar a visualização dos clusters.")

# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# --reload é útil para desenvolvimento, pois reinicia o servidor quando o código muda.
