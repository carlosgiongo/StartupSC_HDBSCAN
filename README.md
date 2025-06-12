
## Estudo - HDBSCAN em startups aprovadas da StartupSC

Este repositório contém os principais objetos do estudo do uso da técnica de clusterização HDBSCAN nas startups aprovadas no programa de fomento StartupSC.

Para o deploy da ferramenta, é necessário que seja alocado o arquivo index.html dentro de um servidor web que possibilite seu consumo, como APACHE ou NGINX.

Para o uso do recurso que possibilita a inferencia de novos dados textuais de uma nova organização para verificação no plano tridimensional, é necessário a execução da ferramenta da pasta "api" com o serviço da biblioteca Uvicorn.  

```bash
  cd api
  python3 -m venv venv
  source venv/bin/activate
  pip3 install -r requirements.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8123
```

O arquivo index.html já aponta para o serviço em http://localhost:8123/visualizacao-clusters

