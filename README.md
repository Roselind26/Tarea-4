# Predicción de Cáncer de Mama

Este proyecto utiliza un modelo de **regresión logística** entrenado con el dataset **Breast Cancer Wisconsin (Diagnostic)** para predecir si un tumor es benigno o maligno.

## Variables de Entrada del Modelo

El modelo espera las siguientes variables de entrada, todas de tipo **float**:

- **radius_mean**: Promedio del radio del tumor
- **texture_mean**: Promedio de la textura del tumor
- **perimeter_mean**: Promedio del perímetro del tumor
- **area_mean**: Promedio del área del tumor
- **smoothness_mean**: Promedio de la suavidad del tumor
- **compactness_mean**: Promedio de la compactación del tumor
- **concavity_mean**: Promedio de la concavidad del tumor
- **concave_points_mean**: Promedio de los puntos cóncavos en el tumor
- **symmetry_mean**: Promedio de la simetría del tumor
- **fractal_dimension_mean**: Promedio de la dimensión fractal del tumor
- **(Y otras variables de la misma forma)**

## Salida del Modelo

El modelo devolverá uno de los siguientes resultados:

- **Benigno**: El tumor es considerado benigno (no cancerígeno).
- **Maligno**: El tumor es considerado maligno (cancerígeno).

La salida será una cadena con el valor:

- `"Benigno "`
- `"Maligno "`

## Ejemplo de uso

1. Ingrese los valores en los campos proporcionados.
2. El modelo procesará los datos y devolverá una predicción (Benigno o Maligno).
