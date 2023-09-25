import numpy as np
from numpy import dot, ndarray
import numpy.linalg as linalg
import math
from filterpy.kalman import predict, update


n_roll_real = 3
n_pitch_real = 3

m_roll_real = 0.1
m_pitch_real = 0.1
m_yaw_real = 0.1
i = 0
count = 0
x_barra = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
rateCalibrationRoll = [-1.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
rateCalibrationPitch = [1.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
rateCalibrationYaw = [-0.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

acelCalibrationRoll = [1.03, 0.94, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
acelCalibrationPitch = [0.99, 1.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
acelCalibrationYaw = [1.07, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# dados = [1,2,3,4,5,6,7,8,9,2,1,2,3,4]
# Impelmentar busca de arquivos e leitura (pandas)
# Criar for com loop para pegar a cada 7 dados fornecidos pelo cartão SD (Traduzir do código C#)
# Multiplicação de matrizes feita, agora ver transformação para quaternion e depois manipulação das strings

c = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

r = np.array([[n_roll_real, 0.0],
              [0.0, n_pitch_real]])

x = np.array([[0.0],
              [0.0],
              [0.0],
              [0.0],
              [0.0],
              [0.0]])

u = np.array([[1.0],
              [2.0],
              [3.0]])

a_shape = (6, 6)
P = np.zeros(a_shape)

with open('Teste.txt') as f:
    contents = f.read()
    dados = contents.split(',')
    print(dados)

for i in range(0, len(dados), 7):
    Ti = float(dados[i + 6])

    q = np.array([[m_roll_real * Ti * Ti, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, m_pitch_real * Ti * Ti, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, m_yaw_real * Ti * Ti, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, m_roll_real, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, m_pitch_real, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, m_yaw_real]])

    b = np.array([[Ti, 0.0, 0.0],
                  [0.0, Ti, 0.0],
                  [0.0, 0.0, Ti],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])

    a = np.array([[1.0, 0.0, 0.0, -Ti, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, -Ti, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, -Ti],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    #nstate_nobs = (6, 2)
    #nobs_nobs = (2, 2)

    #S = np.zeros(nobs_nobs)
    #k = np.zeros(nstate_nobs)

    # Recebimento -> ACCX, ACCY, ACCZ, GYRX, GYRY, GYRXZ e Tempo
    # dados[i], dados[i+1], dados[i+2], dados[i+3], dados[i+4], dados[i+5], dados[i+6]

    # AngleRoll = atan(AccY / sqrt(AccX * AccX + AccZ * AccZ)) * 1 / (3.14159265 / 180);
    angleRoll = math.atan((float(dados[i + 1]) - acelCalibrationPitch[count]) / math.sqrt(
        ((float(dados[i]) - acelCalibrationRoll[count]) * (float(dados[i]) - acelCalibrationRoll[count])) + (
                (float(dados[i + 2]) - acelCalibrationYaw[count]) * (
                float(dados[i + 2]) - acelCalibrationYaw[count])))) * (1 / (math.pi / 180))

    # AnglePitch = -atan(AccX/sqrt(AccY*AccY + AccZ*AccZ))*1/(3.14159265/180);
    anglePitch = -math.atan((float(dados[i]) - acelCalibrationRoll[count]) / math.sqrt(
        ((float(dados[i + 1]) - acelCalibrationPitch[count]) * (float(dados[i + 1]) - acelCalibrationPitch[count])) + (
                (float(dados[i + 2]) - acelCalibrationYaw[count]) * (
                float(dados[i + 2]) - acelCalibrationYaw[count])))) * (1 / (math.pi / 180))

    # rateRoll = (Convert.ToDouble(list[i + 3]) / 131) - rateCalibrationRoll[count];
    # ratePitch = (Convert.ToDouble(list[i + 4]) / 131) - rateCalibrationPitch[count];
    # rateYaw = (Convert.ToDouble(list[i + 5]) / 131) - rateCalibrationYaw[count];

    rateRoll = (float(dados[i + 3]) / 131) - rateCalibrationRoll[count]
    ratePitch = (float(dados[i + 4]) / 131) - rateCalibrationPitch[count]
    rateYaw = (float(dados[i + 5]) / 131) - rateCalibrationYaw[count]

    # y = np.matmul(a, x)
    # z = np.matmul(b, u)
    # p = y + z

    # print(np.array(y))
    # print(np.array(z))
    x, P = predict(x, P, a, q, u, b)
    print("----------------Predict-----------------")
    print('x =', x)
    print('P =', P)

    S = dot(dot(c, P), c.T) + r

    is_nosingular = bool(S.inv(S))

    k = dot(dot(P, c.T), S)

    if is_nosingular:
        x, P = update(x, P, 1, r, c)
        print("----------------Update-----------------")
        print('x =', x)
        print('P =', P)



    # print("Array %d" % (count + 1))
    # print(np.array(p))
    # x_barra[count] = np.array(p)

    count += 1

# print("Array xbarra: ")
# print(x_barra[0])
