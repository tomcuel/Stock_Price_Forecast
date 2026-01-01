import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

def create_complex_time_series(n_samples=2000, seq_len=50, n_features=3, seed=42):
    np.random.seed(seed)
    t = np.arange(n_samples + seq_len)
    series = []
    for f in range(n_features):
        freq = np.random.uniform(0.01, 0.1)
        amp = np.random.uniform(0.5, 2.0)
        wave = amp * np.sin(2 * np.pi * freq * t) + amp/2 * np.cos(2 * np.pi * freq*0.5 * t)
        trend = t * np.random.uniform(0.0005, 0.002)
        season = 0.5 * np.sin(2 * np.pi * t / np.random.randint(30, 100))
        noise = np.random.normal(0, 0.2, len(t))
        spikes = np.zeros(len(t))
        spike_idx = np.random.choice(len(t), size=int(0.01*len(t)), replace=False)
        spikes[spike_idx] = np.random.uniform(1, 3, len(spike_idx))
        series.append(wave + trend + season + noise + spikes)
    series = np.stack(series, axis=-1)
    X, y = [], []
    for i in range(n_samples):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len, 0])
    X = np.array(X)  # (n_samples, seq_len, n_features)
    y = np.array(y)  # (n_samples,)
    return X, y

SEQ_LEN = 50
N_FEATURES = 3
X, y = create_complex_time_series(n_samples=2000, seq_len=SEQ_LEN, n_features=N_FEATURES)

def build_encoder_decoder_dataset(X, y, dec_len=None):
    N, seq_len, n_features = X.shape
    if dec_len is None:
        dec_len = seq_len
    X_enc = X.copy()                       # (N, seq_len, n_features)
    X_dec = np.zeros_like(X)               # initialize
    X_dec[:, 1:, :] = X[:, :-1, :]         # teacher forcing
    X_dec[:, 0, :] = 0                     # start token
    Y = y.reshape(-1, 1)                   # (N, 1)
    return X_enc, X_dec, Y

X_enc, X_dec, Y = build_encoder_decoder_dataset(X, y, dec_len=SEQ_LEN)
train_size = int(0.8 * len(X))
X_enc_train, X_enc_val = X_enc[:train_size], X_enc[train_size:]
X_dec_train, X_dec_val = X_dec[:train_size], X_dec[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]

class MultiQueryAttention(layers.Layer):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.supports_masking = True
        self.q_proj = layers.Dense(num_heads * key_dim)
        self.k_proj = layers.Dense(key_dim)
        self.v_proj = layers.Dense(key_dim)
        self.out = layers.Dense(num_heads * key_dim)

    def call(self, query, value=None, key=None, mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
        B = tf.shape(query)[0]
        Tq = tf.shape(query)[1]
        Tk = tf.shape(key)[1]
        q = self.q_proj(query)                  # (B, Tq, H*D)
        k = self.k_proj(key)                    # (B, Tk, D)
        v = self.v_proj(value)                  # (B, Tk, D)
        q = tf.reshape(q, (B, Tq, self.num_heads, self.key_dim))
        q = tf.transpose(q, [0, 2, 1, 3])        # (B, H, Tq, D)
        k = k[:, None, :, :]                    # (B, 1, Tk, D)
        v = v[:, None, :, :]                    # (B, 1, Tk, D)
        scores = tf.matmul(q, k, transpose_b=True)  # (B, H, Tq, Tk)
        scores /= tf.sqrt(tf.cast(self.key_dim, tf.float32))
        if mask is not None:
            scores += mask   # broadcasting works if mask is (B, 1, Tq, Tk)
        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, v)                # (B, H, Tq, D)
        out = tf.transpose(out, [0, 2, 1, 3])    # (B, Tq, H, D)
        out = tf.reshape(out, (B, Tq, self.num_heads * self.key_dim))
        return self.out(out)
    
class EncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, use_mqa=False):
        super().__init__()
        if use_mqa:
            self.attn = MultiQueryAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        else:
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, x, mask=None, training=False):
        if isinstance(self.attn, MultiQueryAttention):
            attn_out = self.attn(x, mask=mask)
        else:
            attn_out = self.attn(query=x, value=x, key=x, mask=mask)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x + ffn_output)
    
class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, use_mqa=False):
        super().__init__()
        if use_mqa:
            self.self_attn = MultiQueryAttention(num_heads, d_model // num_heads)
            self.cross_attn = MultiQueryAttention(num_heads, d_model // num_heads)
        else:
            self.self_attn = layers.MultiHeadAttention(num_heads, d_model // num_heads)
            self.cross_attn = layers.MultiHeadAttention(num_heads, d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.dropout3 = layers.Dropout(0.1)

    def call(self, x, enc_out, causal_mask=None, training=False):
        if isinstance(self.self_attn, MultiQueryAttention):
            attn1 = self.self_attn(x, mask=causal_mask)
            attn2 = self.cross_attn(x, enc_out)
        else:
            attn1 = self.self_attn(query=x, value=x, key=x, mask=causal_mask)
            attn2 = self.cross_attn(query=x, value=enc_out, key=enc_out)
        attn1 = self.dropout1(attn1, training=training)
        x = self.norm1(x + attn1)
        attn2 = self.dropout2(attn2, training=training)
        x = self.norm2(x + attn2)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout3(ffn_out, training=training)
        x = self.norm3(x + ffn_out)
        return x
    
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pe = np.zeros_like(angle_rads)
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(pe[None, ...], tf.float32)

class TransformerTimeSeries(tf.keras.Model):
    def __init__(self, enc_len, dec_len,
                 d_model=64, num_heads=4, ff_dim=128,
                 num_enc_layers=2, num_dec_layers=2,
                 use_mqa=False):
        super().__init__()
        self.enc_embedding = layers.Dense(d_model)
        self.dec_embedding = layers.Dense(d_model)
        self.enc_pos = positional_encoding(enc_len, d_model)
        self.dec_pos = positional_encoding(dec_len, d_model)
        self.encoder = [EncoderBlock(d_model, num_heads, ff_dim, use_mqa) for _ in range(num_enc_layers)]
        self.decoder = [DecoderBlock(d_model, num_heads, ff_dim, use_mqa) for _ in range(num_dec_layers)]
        self.out = layers.Dense(1)

    def causal_mask(self, seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)   # Shape (1, 1, seq_len, seq_len)
        return mask[None, None, :, :] * -1e9

    def call(self, inputs, training=False):
        encoder_input, decoder_input = inputs
        x = self.enc_embedding(encoder_input) + self.enc_pos
        for layer in self.encoder:
            x = layer(x, training=training)
        enc_out = x   # (B, enc_len, d_model)
        y = self.dec_embedding(decoder_input) + self.dec_pos
        dec_len = tf.shape(decoder_input)[1]
        mask = self.causal_mask(dec_len)
        for layer in self.decoder:
            y = layer(y, enc_out, causal_mask=mask, training=training)
        return self.out(y)[:, -1:, :]   # last step in sequence
    
