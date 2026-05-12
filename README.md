## Esse projeto é uma adaptação do projeto descrito em Unsupervised anomaly detection in brain MRI: Learning abstract distribution from massive healthy brains

Link: https://www.sciencedirect.com/science/article/pii/S0010482523000756





# Uso docker

Build image with
docker build -t <NAME> .



Run container with:
docker run -it --rm \
  -p 8889:8888 \
  --gpus all \
  --mount type=bind,src="$(pwd)",dst=/home/jovyan/work \
  -e CHOWN_EXTRA="/home/jovyan/work" \
  -e CHOWN_USER=1000 \
  -e CHOWN_GROUP=100 \
  ae_env \
  start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''



  docker run -it --rm \
  -p 8889:8888 \
  --gpus all \
  --mount type=bind,src="$(pwd)",dst=/home/jovyan/work \
  --mount type=bind,src="D:/Backup/Mestrado/dados/",dst=/home/jovyan/imgs \
  -e CHOWN_EXTRA="/home/jovyan/work,/home/jovyan/imgs" \
  -e CHOWN_USER=1000 \
  -e CHOWN_GROUP=100 \
  ae_env \
  start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''


# With mlflow
docker run -it \
  --name ae_env_mlflow \
  -p 8889:8888 \
  -p 5000:5000 \
  --gpus all \
  --mount type=bind,src="$(pwd)",dst=/home/jovyan/work \
  --mount type=bind,src="D:/Backup/Mestrado/dados/",dst=/home/jovyan/imgs \
  -e CHOWN_EXTRA="/home/jovyan/work,/home/jovyan/imgs" \
  -e CHOWN_USER=1000 \
  -e CHOWN_GROUP=100 \
  ae_env \
  start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''


# Run mlflow
mlflow server --host 0.0.0.0 --port 5000