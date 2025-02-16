import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

# Load the Portuguese-English translation dataset
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

print("Number of training examples: {}".format(len(train_examples)))
print("Number of validation examples: {}".format(len(val_examples)))

# Display the first 3 examples using the batch method from tf.data.Dataset. In tensorflow, we do not have dataloaders.
# for pt_examples, en_examples in train_examples.batch(3).take(1):
#  print('> Examples in Portuguese:')
#  for pt in pt_examples.numpy():
#    print(pt.decode('utf-8'))
#  print()

#  print('> Examples in English:')
#  for en in en_examples.numpy():
#    print(en.decode('utf-8'))

# The examples are raw text, where in this case, words are separated from punctuation and each other by spaces.
# Words are also lower case.
# We need now to tokenize this text. We use BERT tokenization for both languages (model loaded from tensorflow tutorials)

tokenizer_model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{tokenizer_model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{tokenizer_model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizer_model_name = tokenizer_model_name + '_extracted/' + tokenizer_model_name
tokenizers = tf.saved_model.load(tokenizer_model_name)

# Data pipeline. Texts are convertex into ragged batches, that are batches containing ragged tensors where each
# element can have a different length. Tokenized representations are also trimmed to be no longer than MAX_TOKENS. 
# Also, it splits the target (english) tokens into inputs and labels. The labels are shifted by one step so that at each
# input locaton the label is the id of the next token. After that, the ragged tensor is convertex to padded dense Tensor.

MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Trim to MAX_TOKENS
    pt = pt[:, :MAX_TOKENS] # This is probably not the best way to work with long sentences, simply following tensorflow tutorial.
    pt = pt.to_tensor(default_value=0) # Pad value set to default_value=0

    en = tokenizers.en.tokenize(en)
    # Trim to MAX_TOKENS
    en = en[:, :MAX_TOKENS] 
    en = en.to_tensor(default_value=0)
    en_inputs = en[:, :-1].to_tensor(default_value=0) # Drop the [END] tokens.
    en_labels = en[:, 1:].to_tensor(default_value=0) # Drop [START] tokens.
    return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis] # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] # (1, depth)

    angle_rates = 1 / (10000**depths) # (1, depth)
    angle_rads = positions*angle_rates # (seq, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) # (seq, depth*2)

    return tf.cast(pos_encoding, dtype=tf.float32)

# Now we define a positional encoding layer that uses the previous positional encoding function based on sin/cos.
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) # Zero is the padding value. 
        # Differently from PyTorch, you can not set which padding value you want to use in the Embedding layer.
        self.pos_encoding = positional_encoding(vocab_size, d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # Basically we are scaling the embeddings why the width of the hidden dimension.
        # it is to avoid embeddings having too small magnitude compared to positional encodings.
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

# Now we define the transformer model.
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=False)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            return_attention_scores=False
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True,
            return_attention_scores=False
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=rate
        )
        self.ffn = FeedForward(d_model, dff, rate) # For a better customization, we could have a 
        # different dropout rate for the feed forward layer.
    
    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x) # Shape `(batch_size, seq_len, d_model)`.
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x # Shape `(batch_size, seq_len, d_model)`.
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        x = self.ffn(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        return x
    
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs

        context = self.encoder(context) # (batch_size, context_len, d_model)
        x = self.decoder(x, context) # (batch_size, target_len, d_model)
        logits = self.final_layer(x) # (batch_size, target_len, target_vocab_size)
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731 We mask the loss manually
            del logits._keras_mask
        except AttributeError:
            pass
        return logits

class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

learning_rate = TransformerScheduler(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(label, pred)
    loss = tf.reduce_sum(loss_object*tf.cast(mask, tf.float32))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask # Masking padded values

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask) # Average loss
    return loss

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)