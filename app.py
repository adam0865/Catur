# =========================================================================
# APLIKASI WEB CATUR DENGAN STREAMLIT
# =========================================================================
# Deskripsi:
# Aplikasi ini memungkinkan pengguna bermain catur melawan AI yang menggunakan
# model Keras (Value Network) yang telah Anda buat sebelumnya.
#
# Cara Menjalankan:
# 1. Pastikan file model 'chess_ppo_model_console.weights.h5' ada di folder ini.
# 2. Jalankan dari terminal: streamlit run app.py
#

import streamlit as st
import chess
import chess.svg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import os
import time

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Catur AI", layout="wide")

# ========================
# 1. PATH MODEL DAN FUNGSI INTI (DARI SKRIP ANDA)
# ========================
MODEL_PATH = "chess_ppo_model_console.weights.h5"

@st.cache_resource  # <-- Dekorator penting untuk performa!
def load_keras_model():
    """
    Membangun arsitektur dan memuat bobot model.
    Menggunakan cache Streamlit agar model tidak dimuat ulang setiap saat.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan di path: {MODEL_PATH}")
        st.stop()

    model = Sequential([
        Flatten(input_shape=(8, 8, 12)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.load_weights(MODEL_PATH)
    return model

def board_to_state(board: chess.Board) -> np.ndarray:
    """Mengonversi papan catur menjadi state yang dapat dibaca model."""
    state = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11
    }
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = i // 8, i % 8
            state[row, col, piece_map[(piece.piece_type, piece.color)]] = 1.0
    return state

def get_ai_move(board: chess.Board, model: keras.Model) -> chess.Move:
    """Logika AI untuk memilih langkah terbaik berdasarkan skor dari model."""
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None

    best_move = None
    # AI (Hitam) mencari skor terendah (posisi terbaik untuk Hitam)
    best_score = np.inf 

    for move in legal_moves:
        temp_board = board.copy()
        temp_board.push(move)
        state = np.expand_dims(board_to_state(temp_board), axis=0)
        score = model.predict(state, verbose=0)[0][0]
        if score < best_score:
            best_score = score
            best_move = move
            
    return best_move if best_move is not None else np.random.choice(legal_moves)


# ========================
# 2. INISIALISASI STATE & UI STREAMLIT
# ========================
st.title("♟️ Catur AI vs Manusia")
st.markdown("Anda bermain sebagai **Putih**. AI bermain sebagai **Hitam**. Masukkan gerakan Anda dalam format UCI (contoh: `e2e4`).")

# Memuat model AI
ai_model = load_keras_model()

# Menggunakan session_state untuk menyimpan kondisi permainan
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "last_move" not in st.session_state:
    st.session_state.last_move = None

# Tata letak halaman menggunakan kolom
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Papan Catur")
    # Menampilkan papan catur sebagai gambar SVG
    board_svg = chess.svg.board(
        board=st.session_state.board, 
        lastmove=st.session_state.last_move,
        size=400
    )
    st.image(board_svg, use_column_width=True)

    if st.button("Mulai Ulang Permainan"):
        st.session_state.board = chess.Board()
        st.session_state.last_move = None
        st.rerun()

# ========================
# 3. LOGIKA PERMAINAN
# ========================
game_over = st.session_state.board.is_game_over()

if game_over:
    st.warning(f"**Permainan Selesai!** Hasil: {st.session_state.board.result()}")
else:
    # Giliran Manusia (Putih)
    if st.session_state.board.turn == chess.WHITE:
        with st.form("move_form"):
            user_move_uci = st.text_input("Giliran Anda (Putih). Masukkan langkah (UCI):", placeholder="e.g., e2e4")
            submitted = st.form_submit_button("Lakukan Gerakan")

        if submitted and user_move_uci:
            try:
                move = chess.Move.from_uci(user_move_uci.strip())
                if move in st.session_state.board.legal_moves:
                    st.session_state.board.push(move)
                    st.session_state.last_move = move
                    st.rerun()
                else:
                    st.error("Langkah tidak valid!")
            except ValueError:
                st.error("Format langkah salah. Gunakan notasi UCI.")
    
    # Giliran AI (Hitam)
    else:
        with st.spinner("AI (Hitam) sedang berpikir..."):
            ai_move = get_ai_move(st.session_state.board, ai_model)
            if ai_move:
                st.session_state.board.push(ai_move)
                st.session_state.last_move = ai_move
                st.rerun()

with col2:
    st.subheader("Informasi Permainan")
    turn_text = "Pemain (Putih)" if st.session_state.board.turn == chess.WHITE else "AI (Hitam)"
    st.write(f"**Giliran:** {turn_text}")
    
    status_text = "Selesai" if game_over else "Berlangsung"
    st.write(f"**Status:** {status_text}")

    st.subheader("Riwayat Langkah (PGN)")
    pgn_string = chess.Board().variation_san(st.session_state.board.move_stack)
    st.text_area("Langkah", value=pgn_string, height=300, disabled=True)
