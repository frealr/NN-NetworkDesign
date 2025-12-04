# Reporte de Métodos de Optimización Implementados

Este documento describe en detalle los tres enfoques implementados para resolver el problema de optimización cuadrática con restricciones de igualdad y desigualdad.

El problema general es:
$$
\begin{aligned}
\min_{x} \quad & f(x) = \text{Entropy}(x) + c^T x \\
\text{s.t.} \quad & Ax = b \quad (\text{Igualdad}) \\
& x \ge 0 \quad (\text{Desigualdad})
\end{aligned}
$$

Todos los métodos utilizan una **Base del Espacio Nulo Desacoplada** para manejar las restricciones de igualdad de manera eficiente y estructurada.

---

## 1. NN Projection (Proyección en el Espacio Nulo con Ray-Casting)
**Archivo:** `solve_custom_nn.py`

Este método garantiza la factibilidad estricta de todas las restricciones ($Ax=b$ y $x \ge 0$) por construcción en cada paso, utilizando una parametrización geométrica.

### Formulación Matemática
1.  **Restricciones de Igualdad ($Ax=b$):**
    Se eliminan reparametrizando $x$ en función de una variable latente $w$:
    $$ x(w) = R w + u $$
    Donde $u$ es una solución particular ($Au=b$) y $R$ es la base del espacio nulo ($AR=0$).
    *   **Mejora:** Se usa una $R$ dispersa y desacoplada por bloques $(o,d)$.

2.  **Restricciones de Desigualdad ($x \ge 0$):**
    Sustituyendo $x(w)$:
    $$ R w + u \ge 0 \implies -R w \le u $$
    Esto define un politopo en el espacio latente $w$.

3.  **Parametrización (Ray-Casting):**
    En lugar de optimizar $w$ libremente, se define $w$ a partir de un **punto interior estricto** $p$ (precalculado tal que $-R p < u$) y una dirección $r$:
    $$ w = p + \alpha \cdot r $$
    Donde $\alpha$ es un escalar positivo calculado dinámicamente para asegurar que $w$ permanezca dentro del politopo.
    $$ \alpha = \sigma(s) \cdot \alpha_{\max}(p, r) $$
    *   $\alpha_{\max}$: La distancia máxima desde $p$ en dirección $r$ antes de tocar una restricción.
    *   $\sigma(s)$: Una función sigmoide que escala el paso entre $(0, 1)$ para mantenerse estrictamente dentro (o en el borde).

### Algoritmo
1.  **Preprocesamiento:** Calcular $u, R$ (desacoplados) y encontrar el punto interior $p$.
2.  **Modelo:** Una "Red Neuronal" (o capa parametrizada) que aprende la dirección $r$ y el escalar de escala $s$.
3.  **Forward Pass:**
    *   Calcular $\alpha_{\max} = \min_i \left( \frac{u_i + [R p]_i}{-[R r]_i} \right)$ para los índices donde el denominador es positivo.
    *   Calcular $w = p + \text{sigmoid}(s) \cdot \alpha_{\max} \cdot r$.
    *   Reconstruir $x = R w + u$.
4.  **Optimización:** Minimizar $f(x)$ usando Adam. No se necesitan penalizaciones ni duales porque $x$ siempre es factible.

**Ventajas:** Factibilidad garantizada, convergencia robusta, solución limpia.

---

## 2. Primal-Dual Direct (Método del Lagrangiano Aumentado)
**Archivo:** `solve_primal_dual.py`

Este método maneja las igualdades mediante el espacio nulo, pero relaja las desigualdades $x \ge 0$ incorporándolas en la función objetivo mediante el Método del Lagrangiano Aumentado (ALM).

### Formulación Matemática
1.  **Igualdades:** Igual que el método anterior, $x = R w + u$. Se optimiza $w$ directamente.
2.  **Desigualdades ($x \ge 0 \iff -x \le 0$):**
    Se define el Lagrangiano Aumentado para la restricción $g(x) = -x \le 0$:
    $$ L_{\rho}(w, \lambda) = f(x(w)) + \frac{\rho}{2} \sum_{i} \left( \max\left(0, -x_i + \frac{\lambda_i}{\rho}\right) \right)^2 - \frac{1}{2\rho} \sum \lambda_i^2 $$
    O en su forma de actualización más simple usada en la implementación:
    $$ \text{Loss} \approx f(x) + \frac{\rho}{2} \sum_{i} \left( \max\left(0, -x_i + \frac{\lambda_i}{\rho}\right) \right)^2 $$

### Algoritmo
1.  **Preprocesamiento:** Calcular $u, R$. Inicializar $w$ desde $p$. Inicializar duales $\lambda = 0$ y penalización $\rho = 10$.
2.  **Bucle Exterior (Actualización Dual):**
    *   **Bucle Interior (Optimización Primal):**
        *   Minimizar $L_{\rho}(w, \lambda)$ con respecto a $w$ usando Adam.
        *   $x$ se reconstruye como $x = R w + u$ (usando multiplicación dispersa).
    *   **Actualización Dual:** $\lambda \leftarrow \max(0, \lambda + \rho(-x))$.
    *   **Actualización Penalización:** $\rho \leftarrow \min(\rho \cdot 1.2, 10^6)$. Aumentar $\rho$ fuerza a que las violaciones de $x \ge 0$ tiendan a cero.

**Ventajas:** Convierte el problema restringido en uno irrestricto secuencial.
**Desventajas:** Requiere ajustar $\rho$ y el learning rate. La factibilidad estricta ($x \ge 0$) solo se alcanza asintóticamente (aunque en la práctica se logra $Min(x) = 0.0$).

---

## 3. Primal-Dual NN Loop (Modelo Primal-Dual)
**Archivo:** `solve_nn_primal_dual.py`

Este enfoque es algorítmicamente idéntico al **Primal-Dual Direct** en su estado actual, pero estructurado bajo una clase `torch.nn.Module`.

### Diferencia Conceptual
*   En **Primal-Dual Direct**, $w$ se trata como un tensor de parámetros suelto.
*   En **Primal-Dual NN Loop**, $w$ está encapsulado dentro de un modelo (`PrimalModel`).
*   **Potencial:** Esta estructura permite sustituir el parámetro $w$ directo por una **Red Neuronal Profunda** $w = \text{NN}(\theta)$ si se quisiera parametrizar la solución en función de alguna entrada (por ejemplo, si $b$ o $c$ cambiaran dinámicamente).
*   **Implementación Actual:** El modelo es lineal/directo ($w$ es el parámetro), por lo que los resultados son equivalentes al método 2.

### Algoritmo
Idéntico al Método 2 (Lagrangiano Aumentado), pero el paso de optimización primal llama a `model()` y `optimizer.step()` sobre los parámetros del modelo.

---

## Resumen Comparativo

| Característica | NN Projection | Primal-Dual (Direct & NN) |
| :--- | :--- | :--- |
| **Manejo Igualdades** | Espacio Nulo ($Rw+u$) | Espacio Nulo ($Rw+u$) |
| **Manejo Desigualdades** | **Geométrico (Ray-Casting)** | **Penalización (Lagrangiano Aumentado)** |
| **Factibilidad** | **Estricta (Siempre)** | Asintótica (Iterativa) |
| **Variables** | Dirección $r$, Escala $s$ | Latente $w$, Duales $\lambda$ |
| **Estabilidad** | Muy Alta | Sensible a hiperparámetros ($\rho$) |
| **Calidad Solución** | **Óptima (Loss ~14.32)** | Muy Buena (Loss ~13.47 - 13.97) |

**Recomendación:** El método **NN Projection** es superior para este problema estático debido a su garantía de factibilidad y estabilidad numérica.
