## 🧠 Entendendo a Rede Neural para Classificação de Pneumonia

Este projeto utiliza uma rede neural profunda (deep neural network) treinada para **classificar radiografias de tórax** como **NORMAL** ou **PNEUMONIA**. A rede é carregada com o TensorFlow/Keras a partir de um arquivo `.hdf5`, que armazena tanto a **estrutura da rede** quanto os **pesos treinados**.

---

### 📦 Carregando o modelo com Keras

O código abaixo carrega o modelo:

```python
model = tf.keras.models.load_model('./models/xray_model.hdf5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
load_model(...): carrega um modelo completo salvo, com arquitetura, pesos e (opcionalmente) configuração de treinamento.
compile(...): recompila o modelo, definindo:
  optimizer='adam': algoritmo de otimização eficiente para redes profundas.
  loss='binary_crossentropy': função de perda usada para classificação binária.
  metrics=['accuracy']: métrica de desempenho usada para avaliação.
```

## 🏗️ Estrutura da Rede Neural

A rede é do tipo Convolutional Neural Network (CNN), ideal para tarefas de visão computacional. Ela é composta por camadas organizadas em blocos:

- Principais camadas:
    - Conv2D: detecta padrões locais da imagem (bordas, manchas, etc.).

    - MaxPooling2D: reduz a dimensão da imagem (downsampling), mantendo as informações mais relevantes.

    - Flatten: transforma os mapas 2D em vetores 1D.

    - Dense: camadas densas (fully connected), responsáveis pela classificação.

    - Dropout (opcional): ajuda a evitar overfitting durante o treinamento.

## 🧮 Coeficientes e Pesos

Durante o treinamento, o modelo aprende pesos (coeficientes) que se ajustam para minimizar o erro entre a saída esperada e a saída prevista. Esses pesos são salvos no arquivo .hdf5 e utilizados para fazer predições em novos exames.

Você pode acessá-los com:

```python
model.summary()         # mostra a estrutura da rede
model.get_weights()     # retorna os coeficientes (pesos) aprendidos
```