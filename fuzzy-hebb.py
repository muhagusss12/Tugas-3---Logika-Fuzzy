#Mengimport module yang digunakan
import numpy as np

#Membuat function hard_limit yang menerima inputan x
#Inputan yang diterima akan menghasilkan 1 jika x>=0 dan -1 jika x<0
def hard_limit(x):
    return np.where(x >= 0, 1, -1)

#Membuat dua set data training beserta target
data_training1 = np.array([[1, -1, -1], [1, -1, -1], [1, 1, 1]])
target1 = np.array([[1]])

data_training2 = np.array([[-1, -1, -1], [1, -1, 1], [1, -1, 1]])
target2 = np.array([[-1]])

#Menghitung bobot dengan mengalikan setiap data training dengan target
def calculate_weights(data_training, target):
    weights = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            weights[i, j] = data_training[i, j] * target

    return weights

#Menghitung bobot final dengan menjumlahkan bobot 1 dan 2
weights1 = calculate_weights(data_training1, target1)
weights2 = calculate_weights(data_training2, target2)
final_weights = weights1 + weights2

#Menampilkan bobot final
print("Bobot yang telah ditraining:")
print(final_weights)

#User menginputkan nilai pada matriks 3x3
user_input = np.array([int(input(f"Masukkan nilai baris ke-{i//3 + 1}, kolom ke-{i%3 + 1} (1 atau -1): ")) for i in range(9)]).reshape(3, 3)

#Menghitung output dengan function hard_limit pada hasil dot matriks input dari pengguna dengan matriks bobot akhir
output = hard_limit(np.dot(user_input.flatten(), final_weights.flatten()))

#Menampilkan hasil output
print("Hasil Output:")
print(output)