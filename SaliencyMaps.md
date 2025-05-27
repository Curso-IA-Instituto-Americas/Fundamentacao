## 🔍 O que são Saliency Maps?

**Saliency Maps** (ou "Mapas de Saliência") são uma técnica de visualização utilizada para entender **quais partes de uma imagem foram mais importantes para a decisão de um modelo de redes neurais**, como por exemplo um classificador de pneumonia a partir de radiografias de tórax.

---

### 🧠 Como funciona?

Durante a inferência, o modelo faz previsões com base em padrões aprendidos. O saliency map calcula **o gradiente da saída do modelo em relação aos pixels da imagem de entrada**. Isso nos mostra **quais pixels influenciaram mais a saída**.

- Se um pixel tem um valor de gradiente alto, significa que ele **afetou bastante a decisão do modelo**.
- Esses valores são transformados em uma imagem de calor (heatmap), destacando as **regiões mais "relevantes" da imagem**.

---

### 📷 Exemplo prático

Se um modelo classifica uma radiografia como "PNEUMONIA", o saliency map pode nos ajudar a ver **quais áreas do pulmão contribuíram para essa decisão**. Isso torna a IA mais explicável e útil para profissionais da saúde.

---

### ✅ Por que usar?

- Para entender **como o modelo toma decisões**
- Para **validar se ele está focando nas regiões corretas**
- Para ajudar na **confiança e explicabilidade** do modelo

---

### 📌 Observação

Saliency Maps mostram **atenção do modelo**, mas não garantem interpretação clínica. Eles são uma ferramenta de suporte à análise, não uma evidência médica.
