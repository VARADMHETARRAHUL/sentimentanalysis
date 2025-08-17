# sentimentanalysis
📝 Sentiment / Sarcasm Classifier

Hey there! 👋
This is a simple NLP project I built using TensorFlow + Keras to detect sarcasm (kinda like sentiment analysis but with extra spice 🌶️).

The dataset comes from the Sarcasm Headlines Dataset, which contains news headlines labeled as sarcastic (1) or not sarcastic (0).

🚀 What this project does

Loads and preprocesses headlines (tokenization + padding).

Trains a neural network with an Embedding layer.

Uses a GlobalAveragePooling1D trick to keep things light.

Classifies if a headline is sarcastic or not.

Exports learned embeddings for visualization in TensorFlow’s Embedding Projector.

⚙️ Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


It’s simple but works pretty well on the dataset! 🎯

📊 Training

Loss function: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

🎉 Example Output

Decoded padded sequence (with OOV tokens + padding):

former <OOV> store clerk sues over secret 'black <OOV> for minority shoppers ? ? ? ? ? ...


Original sentence:

mom starting to fear son's web series closest thing she will have to grandchild


Label:

1  (sarcastic)

🔍 Embedding Visualization

After training, I exported the embeddings (meta.tsv + vecs.tsv).
You can explore them at 👉 TensorFlow Embedding Projector.

Words with similar meaning cluster together — pretty cool to see in action! 😎



💡 Future ideas

Try LSTMs or GRUs instead of average pooling.

Use a bigger pre-trained embedding (like GloVe or Word2Vec).

Fine-tune with Transformer-based models (BERT, DistilBERT).
