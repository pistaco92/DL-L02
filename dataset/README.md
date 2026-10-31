# dataset

Guardar aqui el archivo original del laboratorio (el mismo del Laboratorio 01).

Ejemplos validos:

- `dataset/15 atributos R0-R5.sav`
- `dataset/deterioro_cognitivo.csv`

El codigo base no incluye datos reales y la carpeta esta en `.gitignore`
(solo se versiona este README). Cada grupo debe copiar su archivo aqui.

Columnas esperadas:

- 15 atributos binarios de entrada: `Día`, `Mes`, `Año`, `Estación`, `País`,
  `Ciudad`, `CalleLugar`, `NumeroPiso`, `Miguel2`, `González2`, `Avenida2`,
  `Imperial2`, `A682`, `Caldera2`, `Copiapo2`.
- Objetivos: `GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4`, `GDS_R5`.
- `ID` existe pero **no** es un atributo predictivo.

El orden de las 15 columnas importa: la Conv1D las lee como una secuencia
de largo 15 (ver `docs/01_problema.md`).
