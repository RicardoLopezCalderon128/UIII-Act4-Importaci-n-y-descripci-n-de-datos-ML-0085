import pandas as pd
import matplotlib.pyplot as plt

# 1. Crear diccionario con datos de tractores
tractores_finca = {
    'id_tractor': [201, 202, 203, 204, 205, 206],
    'marca': ['John Deere', 'Massey Ferguson', 'New Holland', 'John Deere', 'Valtra', 'Massey Ferguson'],
    'modelo': ['5075E', '4710', 'T6.145', '6120M', 'A424', '5713'],
    'ano': [2018, 2015, 2020, 2021, 2019, 2017],
    'horas_motor': [3500, 5200, 1800, 1200, 2500, 4000],
    'combustible': ['Diesel', 'Diesel', 'Diesel', 'Diesel', 'Diesel', 'Diesel'],
    'potencia_hp': [75, 100, 145, 120, 100, 110],
    'ultimo_mantenimiento': ['2023-11-01', '2023-10-15', '2023-12-01', '2024-01-05', '2023-11-20', '2023-12-10'],
    'operador_asignado': ['Juan', 'Pedro', 'Luis', 'Carlos', 'Juan', 'Pedro'],
    'estado': ['Operativo', 'En reparacion', 'Operativo', 'Operativo', 'Operativo', 'Operativo']
}

print("Análisis Exploratorio de Datos EDA")
print("(EDA, Exploratory Data Analysis) Lopez Ricardo NC 0085")
print("5. Crear el diccionario con 10 listas y 6 elementos")

# 2. Convertir diccionario en DataFrame
df = pd.DataFrame(tractores_finca)

print("\n--- TABLA ORIGINAL DE TRACTORES ---")
print(df)

# 3. Exportar los datos a CSV
print("+-+-+- Convertir el DataFrame en archivo -+-+-+")

df.to_csv(
    'tractores_finca.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n✅ Archivo CSV creado correctamente")
print("+-+-+-+-+-8. Leer los datos desde el archivo CSV-+-+-+-+-+")
print("tractores_finca.csv")

# 4. Leer el archivo CSV
df = pd.read_csv('tractores_finca.csv')

print("\n--- DATOS LEÍDOS DESDE CSV ---")
print(df)

print("+-+-+- 9. Análisis Exploratorio de Datos (EDA) -+-+-+")

# 5. Primeros y últimos registros
print("-+-+-+\n1. PRIMEROS REGISTROS (Head): +-+-+-")
print(df.head(3))

print("\n2. ÚLTIMOS REGISTROS (Tail):")
print(df.tail(2))

# 6. Dimensiones y estructura
print("+-+-+- 6. Dimensiones y estructura -+-+-+")
print(df)

print("\n3. DIMENSIONES (Filas, Columnas):", df.shape)
print("\n4. NOMBRES DE COLUMNAS:\n", df.columns)
print("\n5. TIPOS DE DATOS:\n", df.dtypes)

print("\n6. INFORMACIÓN GENERAL:")
df.info()

# 7. Estadísticas y limpieza
print("-+-+-+-+\n7. ESTADÍSTICAS DESCRIPTIVAS:+-+-+-+-+-")
print(df.describe())

print("\n8. VALORES FALTANTES (Nulls):")
print(df.isnull().sum())

print("\n9. REGISTROS DUPLICADOS:")
print(df.duplicated().sum())

print("\n10. VALORES ÚNICOS POR COLUMNA:")
print(df.nunique())
print("Marcas únicas:", df['marca'].unique())

# 8. Análisis específico
print(df)
print("--------------------------------------------------------")

print("\n11. TRACTORES POR MARCA:")
print(df['marca'].value_counts())

print("\n12. PROMEDIO DE POTENCIA (HP):", df['potencia_hp'].mean())
print("13. POTENCIA MÁXIMA (HP):", df['potencia_hp'].max())
print("14. POTENCIA MÍNIMA (HP):", df['potencia_hp'].min())

# 9. Ordenamiento, filtrado y agrupación
print(" 9. Ordenamiento, filtrado y agrupación")
print("--------------------------------------------------------")

print("\n15. TRACTORES ORDENADOS POR HORAS DE MOTOR (Mayor a menor):")
print(df.sort_values(by='horas_motor', ascending=False))

print("\n16. TRACTORES CON POTENCIA MAYOR A 100 HP:")
filtro = df[df['potencia_hp'] > 100]
print(filtro)

print("\n17. RESUMEN POR MARCA:")
resumen = df.groupby('marca').agg(
    tractores=('id_tractor', 'count'),
    potencia_promedio=('potencia_hp', 'mean'),
    horas_motor_maximas=('horas_motor', 'max')
)
print(resumen)

# 18. Gráfica de barras: Tractores por marca
conteo = df['marca'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(conteo.index, conteo.values, color='skyblue')
plt.title('Tractores por marca')
plt.xlabel('Marca')
plt.ylabel('Cantidad de tractores')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 19. Gráfica de barras: Horas de motor por tractor
plt.figure(figsize=(8, 5))
plt.bar(df['modelo'], df['horas_motor'], color='lightgreen')
plt.title('Horas de motor por tractor')
plt.xlabel('Modelo')
plt.ylabel('Horas de motor')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 20. Histograma: Distribución de años
plt.figure(figsize=(8, 5))
plt.hist(df['ano'], bins=5, color='orange', edgecolor='black')
plt.title('Distribución de años de los tractores EDA Lopez Ricardo 0085')
plt.xlabel('Año')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 21. Gráfica de dispersión: Potencia vs Horas de motor
plt.figure(figsize=(8, 5))
plt.scatter(df['potencia_hp'], df['horas_motor'], color='red', s=100)
plt.title('Relación entre potencia y horas de motor EDA Lopez Ricardo 0085')
plt.xlabel('Potencia (HP)')
plt.ylabel('Horas de motor')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 22. Diagrama de caja: Distribución de horas de motor
plt.figure(figsize=(7, 5))
plt.boxplot(df['horas_motor'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Distribución de horas de motor (Boxplot) Ricardo Lopez 0085')
plt.ylabel('Horas de motor')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 23. Matriz de correlación
correlacion = df.corr(numeric_only=True)

print("\n--- MATRIZ DE CORRELACIÓN ---")
print(correlacion)

plt.figure(figsize=(9, 6))
plt.imshow(correlacion, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(correlacion.columns)), correlacion.columns, rotation=45)
plt.yticks(range(len(correlacion.columns)), correlacion.columns)
plt.title('Matriz de correlación de variables numéricas EDA Lopez Calderon 0085')
plt.tight_layout()
plt.show()
