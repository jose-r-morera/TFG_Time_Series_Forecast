sudo docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=19 timescale/timescaledb-ha:pg17

## Docker
**Detener**: sudo docker stop timescaledb  
**Encender**: sudo docker start timescaledb
## Para conectar
psql -d "postgres://postgres:19@localhost/postgres"

CREATE TABLE grafcan_punta_hidalgo (
    ts TIMESTAMPTZ NOT NULL PRIMARY KEY,
    air_temperature DOUBLE PRECISION,
    atmospheric_pressure DOUBLE PRECISION,
    relative_humidity DOUBLE PRECISION
);

SELECT create_hypertable('grafcan_punta_hidalgo', by_range('ts'));

CREATE TABLE openmeteo_punta_hidalgo_icon (
    ts TIMESTAMPTZ NOT NULL PRIMARY KEY,
    air_temperature DOUBLE PRECISION,
    atmospheric_pressure DOUBLE PRECISION,
    relative_humidity DOUBLE PRECISION
);

SELECT create_hypertable('openmeteo_punta_hidalgo_icon', by_range('ts'));