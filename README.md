# Predicción de series temporales mediante técnicas de aprendizaje profundo 

## Estaciones GrafCan
[Mapa](http://visor.grafcan.es/visorweb/default.php?svc=svcMeteo)  
[API](https://sensores.grafcan.es/)

### Arona
Estación CL:	**44** - MTD3016CP (SN: 0459)
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:4164)
Localización:	Tenerife, Arona, Rasca (35 m) - Rasca en Arona (Tenerife)
Coordenadas: -16.683871771694157, 28.005849724777484

### La Orotava
Estación MP:	**22** - MTD3016CP (SN: 0399)
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:3713)
Localización:	Tenerife, La Orotava, Camino de Chasna (812 m) - Centro Usos Múltiples Chasna (Camino de Chasna) en La Orotava (Tenerife)  
Coordenadas: -16.528673321794106,  28.367153388517114

### La Laguna
Estación MP:	**9** - MTD3016CP (SN: 0386)  
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:3700)  
Localización:	Tenerife, San Cristóbal de La Laguna, La Cuesta (350 m) - Centro Ciudadano La Cuesta (La Cuesta) en San Cristóbal de La Laguna (Tenerife)  
Coordenadas: -16.292337071782708, 28.467173667455896

### Punta del Hidalgo
Estación CL:	49 - MTD3016CP (SN: 0461)
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:4166)
Localización:	Tenerife, San Cristóbal de La Laguna, La Punta del Hidalgo (54 m) - La Punta del Hidalgo (Tenerife)
    "type": "Point",
    "coordinates": [
      -16.32573868031188,
      28.56950535030508
    ]
-------------------------------------------------
## Validación

### Santa Cruz de Tenerife
Estación MP:	**56** - MTD3016CP
Localización:	  	Tenerife, Santa Cruz de Tenerife, Polígono Costa Sur (92 m) - GRAFCAN (Tenerife)
Coordenadas: -16.267797059024, 28.453829218127

### Garachico Faltan 20 días 4/5 2023 (usamos datos 2024/2025)
18 - MTD3016CP (SN: 0395)
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:3709)
Localización:	Tenerife, Garachico, La Montañeta (922 m) - Centro Cultural La Montañeta (La Montañeta) en Garachico (Tenerife)
  "type": "Point",
    "coordinates": [
      -16.75689429313361,
      28.340361823312847
    ]
  }

### Opcion extra: se podrían usar los cristianos con datos desde mayo 2024 a mayo 2025

-----------------------  

## Descartados
### Santiago del Teide A  ?? (Faltan ~10 días consecutivos) + mucho error con OpenMeteo
Estación MP: **3** - MTD3016CP (SN: 0380)
Localización: Tenerife, Santiago del Teide, Tamaimo (574 m) - Casa Juventud Tamaimo (Tamaimo) en Santiago del Teide (Tenerife)

    "coordinates": 
      -16.819303793235154,
      28.26829646942575

### Santiago del Teide B (Faltan ~10 días consecutivos)  + mucho error con OpenMeteo
46 - MTD3016CP (SN: 0454)
Descripción: 	Geonica Data Acquisition Station METEODATA-3016CP (ID:4159)
Localización: 	Tenerife, Santiago del Teide, CEIP José Esquivel (46 m) - CEIP José Esquivel en Santiago del Teide (Tenerife)
  "id": 46,
  "name": "Tenerife, Santiago del Teide, CEIP José Esquivel (46 m)",
  "description": "CEIP José Esquivel en Santiago del Teide (Tenerife)",
  "encodingType": "application/vnd.geo+json",
  "location": {
    "type": "Point",
    "coordinates": [
      -16.838161919121422,
      28.23341202494747
    ]
  }

### Buenavista del Norte - Muchos datos faltantes y error con OpenMeteo
**50** - MTD3016CP (SN: 0462)
Tenerife, Buenavista del Norte, Punta Teno (64 m) - Punta Teno (Tenerife)
"id": 50,
  "name": "Tenerife, Buenavista del Norte, Punta Teno (64 m)",
  "description": "Punta Teno (Tenerife)",
  "encodingType": "application/vnd.geo+json",
  "location": {
    "type": "Point",
    "coordinates": [
      -16.90852984008341,
      28.356378540693843
    ]
  }


### Adeje Faltan 20 días (11/12 2024) + mucho error con OpenMeteo
Estación MP:	54 - MTD3016CP (SN: 0447)
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:4152)
Localización:	Tenerife, Adeje, Tijoco de arriba (965 m) - Tijoco de arriba en Adeje (Tenerife)
  "coordinates": [
      -16.73359612528364,
      28.169011311312257
    ]
### Los Cristianos --> Faltan muchos días 2023 y 10 días 3 al 4/24
Estación MP:	**47** - MTD3016CP (SN: 0455)  
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:4160)  
Localización:	Tenerife, Arona, IES Los Cristianos (25 m) - IES Los Cristianos en Arona (Tenerife)  
Coordenadas: -16.716917942488553, 28.057623302788404

### El Rosario
Estación CL:	51 - MTD3016CP (SN: 0409)
Descripción:	Geonica Data Acquisition Station METEODATA-3016CP (ID:3724)
Localización:	Tenerife, El Rosario, El Chorrillo (234 m) - El Chorrillo (Tenerife)
   "coordinates": [
      -16.32394055672787,
      28.41041988307398
    ]

## Modelos meteorológicos 
Openmeteo
models=meteofrance_arpege_europe
&models=icon_global

NOTA: Solo existen registros desde el 24/11/2022


OPCIONes:

- Cristianos desde 10/04/2024 (o más adelante)
- Garchico desde feb 2024 (o mas adelante)
- Teno          "

(Santiago descartado por inestable)

