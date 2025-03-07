sudo docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=19 timescale/timescaledb-ha:pg17

## Docker
**Detener**: sudo docker stop timescaledb  
**Encender**: sudo docker start timescaledb
## Para conectar
psql -d "postgres://postgres:19@localhost/postgres"