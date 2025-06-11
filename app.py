# =========================================================================
# SCRIPT UNTUK MENGUJI MODEL CATUR DALAM PERMAINAN SESUNGGUHNYA
# =========================================================================
#
# Deskripsi:
# Skrip ini mengintegrasikan model Keras yang telah Anda buat ke dalam
# sebuah aplikasi catur interaktif di konsol.
#
# Cara Menjalankan:
# 1. Pastikan file model Anda (chess_ppo_model_console.weights.h5)
#    berada di folder yang sama dengan skrip ini.
# 2. Install library yang dibutuhkan:
#    pip install tensorflow "python-chess==1.999"
# 3. Jalankan skrip ini dari terminal:
#    python nama_file_ini.py
#

import chess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import os
import time

# ========================
# 1. Set Path dan Konstanta
# ========================
MODEL_PATH = "chess_ppo_model_console.weights.h5"

# ========================
# 2. Fungsi Model dan Helper (Sesuai dengan kode Anda)
# ========================
def build_model():
    """
    Fungsi ini SAMA PERSIS dengan yang Anda berikan.
    Membangun arsitektur model Sequential.
    """
    model = Sequential([
        Flatten(input_shape=(8, 8, 12)),  # Input dari representasi papan
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')    # Output skor untuk posisi papan
    ])
    # Kita tidak perlu compile model saat hanya ingin melakukan prediksi (inference)
    return model

def board_to_state(board: chess.Board) -> np.ndarray:
    """
    Mengonversi objek papan dari python-chess menjadi
    representasi tensor (8, 8, 12) yang bisa dibaca oleh model Anda.
    """
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

# ========================
# 3. Logika AI untuk Memilih Langkah
# ========================
def get_ai_move(board: chess.Board, model: keras.Model) -> chess.Move:
    """
    Fungsi utama AI untuk memilih langkah terbaik.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = None
    # Jika giliran AI adalah Putih (WHITE), ia mencari skor tertinggi.
    # Jika giliran AI adalah Hitam (BLACK), ia mencari skor terendah.
    best_score = -np.inf if board.turn == chess.WHITE else np.inf

    # Iterasi melalui semua langkah yang legal
    for move in legal_moves:
        # 1. Buat salinan papan dan coba lakukan langkah
        temp_board = board.copy()
        temp_board.push(move)

        # 2. Ubah papan hasil langkah menjadi state untuk model
        state = board_to_state(temp_board)
        state = np.expand_dims(state, axis=0) # Tambah dimensi batch

        # 3. Dapatkan skor dari model Anda
        score = model.predict(state, verbose=0)[0][0]

        # 4. Bandingkan skor untuk menemukan langkah terbaik
        if board.turn == chess.WHITE: # AI (putih) ingin memaksimalkan skor
            if score > best_score:
                best_score = score
                best_move = move
        else: # AI (hitam) ingin meminimalkan skor
            if score < best_score:
                best_score = score
                best_move = move

    # Fallback jika tidak ada langkah yang dipilih (seharusnya tidak terjadi)
    return best_move if best_move is not None else np.random.choice(legal_moves)

# ========================
# 4. Visualisasi dan Game Loop
# ========================
def visualize_board(board: chess.Board):
    """Menampilkan papan catur dalam format teks yang mudah dibaca."""
    print("\n  a b c d e f g h")
    print(" +-----------------+")
    for i in range(7, -1, -1):
        print(f"{i+1}|", end=" ")
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            print(piece.symbol() if piece else ".", end=" ")
        print(f"|{i+1}")
    print(" +-----------------+")
    print("  a b c d e f g h\n")

def play_game(model: keras.Model):
    """Loop utama untuk permainan interaktif."""
    board = chess.Board()
    print("=" * 40)
    print("Selamat Datang! Anda akan bermain catur.")
    print("Anda bermain sebagai PUTIH.")
    print("Masukkan gerakan dalam notasi UCI (contoh: e2e4).")
    print("=" * 40)

    while not board.is_game_over():
        visualize_board(board)

        if board.turn == chess.WHITE: # Giliran Anda
            move_uci = input("Giliran Anda (Putih). Masukkan langkah: ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print(">>> LANGKAH TIDAK VALID! Coba lagi.")
            except ValueError:
                print(f">>> FORMAT '{move_uci}' SALAH! Gunakan notasi UCI.")
        else: # Giliran AI
            print("Giliran AI (Hitam). AI sedang berpikir...")
            start_time = time.time()
            ai_move = get_ai_move(board, model)
            end_time = time.time()
            print(f"AI memilih: {ai_move.uci()} (Waktu berpikir: {end_time - start_time:.2f} detik)")
            if ai_move:
                board.push(ai_move)

    # Permainan Selesai
    print("\n" + "="*30)
    print("PERMAINAN SELESAI!")
    visualize_board(board)
    print(f"Hasil: {board.result()}")


if _name_ == "_main_":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File model '{MODEL_PATH}' tidak ditemukan!")
        print("Harap jalankan skrip pelatihan Anda terlebih dahulu untuk membuat file ini.")
    else:
        print("Model ditemukan. Memuat bobot...")
        # Membangun dan memuat model Anda
        game_model = build_model()
        game_model.load_weights(MODEL_PATH)
        print("Model berhasil dimuat. Siap bermain.")
        
        # Memulai permainan
        play_game(game_model)