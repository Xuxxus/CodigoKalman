import numpy as np
from numpy import dot
import math
from filterpy.kalman import predict, update
import matplotlib.pyplot as plt

float_formartter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formartter})

xbarra = []


def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.

  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

  return [qw, qx, qy, qz]

def getCofactor(mat, temp, p, q, n):
    i = 0
    j = 0

    # Looping for each element
    # of the matrix
    for row in range(n):

        for col in range(n):

            # Copying into temporary matrix
            # only those element which are
            # not in given row and column
            if (row != p and col != q):

                temp[i][j] = mat[row][col]
                j += 1

                # Row is filled, so increase
                # row index and reset col index
                if (j == n - 1):
                    j = 0
                    i += 1


# Recursive function for
# finding determinant of matrix.
# n is current dimension of mat[][].
def determinantOfMatrix(mat, n):
    D = 0  # Initialize result

    # Base case : if matrix
    # contains single element
    if (n == 1):
        return mat[0][0]

    # To store cofactors
    temp = [[0 for x in range(n)]
            for y in range(n)]

    sign = 1  # To store sign multiplier

    # Iterate for each
    # element of first row
    for f in range(n):
        # Getting Cofactor of mat[0][f]
        getCofactor(mat, temp, 0, f, n)
        D += (sign * mat[0][f] *
              determinantOfMatrix(temp, n - 1))

        # terms are to be added
        # with alternate sign
        sign = -sign
    return D


def isInvertible(mat, n):
    if (determinantOfMatrix(mat, n) != 0):
        return True
    else:
        return False



n_roll_real = 3
n_pitch_real = 3

m_roll_real = 0.1
m_pitch_real = 0.1
m_yaw_real = 0.1
i = 0
x_barra = []
tempo = []

rateCalibrationRoll = [-0.13, -2.85, 1.69, -4.2, 7.55, -3.95, 2.65, -26.7, -1.0, -3.65, 0.0, 0.0, 0.0]
rateCalibrationPitch = [0.83, 1.68, -0.89, 2.45, 1.07, -2.74, -3.26, -1.29, 0.72, -0.31, 0.0, 0.0, 0.0]
rateCalibrationYaw = [0.6, 0.53, 0.24, -1.15, 1.87, -0.34, -0.59, 0.65, 0.62, 0.45, 0.0, 0.0, 0.0]

'''
rateCalibrationRoll = [-3.95, -1.0, 1.69, -4.2, 7.55, -3.95, 2.65, -26.7, -1.0, -3.65, 0.0, 0.0, 0.0]
rateCalibrationPitch = [-2.74, 0.72, -0.89, 2.45, 1.07, -2.74, -3.26, -1.29, 0.72, -0.31, 0.0, 0.0, 0.0]
rateCalibrationYaw = [-0.34, 0.62, 0.24, -1.15, 1.87, -0.34, -0.59, 0.65, 0.62, 0.45, 0.0, 0.0, 0.0]'''


acelCalibrationRoll = [1.04, 1.03, 1.04, 1.03, 1.0, 1.03, 1.06, 1.04, 1.03, 0.96, 0.0, 0.0, 0.0]
acelCalibrationPitch = [0.98, 0.98, 0.99, 1.0, 0.99, 1.0, 0.99, 0.98, 0.99, 1.01, 0.0, 0.0, 0.0]
acelCalibrationYaw = [0.95, 0.94, 0.94, 0.95, 0.93, 0.99, 0.74, 1.0, 0.95, 0.89, 0.0, 0.0, 0.0]

'''
acelCalibrationRoll = [1.03, 1.03, 1.04, 1.03, 1.0, 1.03, 1.06, 1.04, 1.03, 0.96, 0.0, 0.0, 0.0]
acelCalibrationPitch = [1.0, 0.99, 0.99, 1.0, 0.99, 1.0, 0.99, 0.98, 0.99, 1.01, 0.0, 0.0, 0.0]
acelCalibrationYaw = [0.99, 0.95, 0.94, 0.95, 0.93, 0.99, 0.74, 1.0, 0.95, 0.89, 0.0, 0.0, 0.0]'''

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

nstate_nobs = (6, 2)
nobs_nobs = (2, 2)
n_z = (2,1)

S = np.zeros(nobs_nobs)
k = np.zeros(nstate_nobs)
z = np.zeros(n_z)

with open('/content/drive/MyDrive/TCC/Kalman/take1.txt') as f:
    contents = f.read()
    dados = contents.split(',')
    print(dados)
    print(len(dados))

n_sensor = int(input("Insira o número de sensores: "))
nome_sensores = []
for i in range(n_sensor):
    nome_sensores.append(str(input("Nome do sensor %d: " %(i+1))))
    xbarra.append([])

print(len(dados))
count = 0
for i in range(0, len(dados), 7):

    TT = float(dados[i+6])
    if i < 7*n_sensor:
      Ti = float(dados[i+6])
    if i > 7*n_sensor:
      Ti =  (float(dados[i+6]) - float(dados[i+6-(7*n_sensor)]))

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

    dados[i] = (float(dados[i ]) - (acelCalibrationRoll[count] -1)) #altereiiiiiiiii
    dados[i + 1] = (float(dados[i + 1]) - (acelCalibrationPitch[count] -1))  #altereiiiiiiiii
    dados[i + 2] =  (float(dados[i + 2]) - (acelCalibrationYaw[count] -1)) #altereiiiiiiiii

    # Recebimento -> ACCX, ACCY, ACCZ, GYRX, GYRY, GYRXZ e Tempo
    # dados[i], dados[i+1], dados[i+2], dados[i+3], dados[i+4], dados[i+5], dados[i+6]

    # AngleRoll = atan(AccY / sqrt(AccX * AccX + AccZ * AccZ)) * 1 / (3.14159265 / 180);
    angleRoll = math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))


    '''math.atan((float(dados[i + 1]) - acelCalibrationPitch[count]) / math.sqrt(
        ((float(dados[i]) - acelCalibrationRoll[count]) * (float(dados[i]) - acelCalibrationRoll[count])) + (
                (float(dados[i + 2]) - acelCalibrationYaw[count]) * (
                float(dados[i + 2]) - acelCalibrationYaw[count]))))'''

    # AnglePitch = -atan(AccX/sqrt(AccY*AccY + AccZ*AccZ))*1/(3.14159265/180);
    anglePitch = -math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
            ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))

    '''-math.atan((float(dados[i]) - acelCalibrationRoll[count]) / math.sqrt(
        ((float(dados[i + 1]) - acelCalibrationPitch[count]) * (float(dados[i + 1]) - acelCalibrationPitch[count])) + (
                (float(dados[i + 2]) - acelCalibrationYaw[count]) * (
                float(dados[i + 2]) - acelCalibrationYaw[count])))) #* (1 / (math.pi / 180)'''

    #angleYaw = (math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))

    z[0] = angleRoll
    z[1] = anglePitch
    #z[2] = angleYaw

    # rateRoll = (Convert.ToDouble(list[i + 3]) / 131) - rateCalibrationRoll[count];
    # ratePitch = (Convert.ToDouble(list[i + 4]) / 131) - rateCalibrationPitch[count];
    # rateYaw = (Convert.ToDouble(list[i + 5]) / 131) - rateCalibrationYaw[count];

    rateRoll = (float(dados[i + 3]) / 131) - rateCalibrationRoll[count]
    ratePitch = (float(dados[i + 4]) / 131) - rateCalibrationPitch[count]
    rateYaw = (float(dados[i + 5]) / 131) - rateCalibrationYaw[count]

    u[0] = rateRoll
    u[1] = ratePitch
    u[2] = rateYaw

    if (i//7)%(n_sensor) == 0 and i != 0:

        tempo.append(TT)
        count = 0
    # y = np.matmul(a, x)
    # z = np.matmul(b, u)
    # p = y + z

    # print(np.array(y))
    # print(np.array(z))
    x, P = predict(x, P, a, q, u, b)
    '''print("----------------Predict-----------------")
    print('x =', x)
    print('P =', P)'''

    S = dot(dot(c, P), c.T) + r

    is_nosingular = determinantOfMatrix(S, 2) #Arrumar esse boolean

    k = dot(dot(P, c.T), S)

    if is_nosingular:
        x, P = update(x, P, z, r, c)
        #print("----------------Update-----------------")
        #print('x =', x)
        #print('P =', P)
        xbarra[count].append(get_quaternion_from_euler(x[2][0],x[1][0],x[0][0]))
        #xbarra[count].append(x_barra)
        #x_barra = []


    count += 1
    print(count)
    if i == 0:
        tempo.append(TT)
        print("Ângulo de euler inicial: " + str(x[0][0]) + str(x[1][0]) + str(x[2][0]))
    # print("Array %d" % (count + 1))
    # print(np.array(p))
    #x_barra[count] = np.array(p)

#print("Xbarra: ",x_barra)
# print("Array xbarra: ")
# print(x_barra[0])
print(len(tempo))
print('----')
print(len(xbarra[0]))
print(xbarra)
#print(x_barra)

'''
for k in range(2):
    for i in range(len(xbarra[0])-1):
        for j in range(4):
            xbarra[k][i][j] = "%.16f" % xbarra[k][i][j]
'''

print(xbarra)

#time.sleep(10)

f1 = open('posicao_inicial.sto', 'w')
f1.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4\nendheader\n")
f1.write('time')
for l in range(len(nome_sensores)):
    f1.write('\t' + nome_sensores[l])
f1.write("\n")
f1.write(str(tempo[0]))
for n in range(n_sensor):
    f1.write("\t" + str(xbarra[n][0][0]) + "," + str(xbarra[n][0][1]) + "," + str(xbarra[n][0][2]) + "," + str(xbarra[n][0][3]))

f1 = open('opensimTeste.sto', 'w')
f1.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4\nendheader\n")
f1.write('time')
for l in range(len(nome_sensores)):
    f1.write('\t' + nome_sensores[l])
f1.write("\n")
for k in range(0, len(xbarra[0])-1, 1):
    f1.write(str(tempo[k]))
    for j in range(0, n_sensor, 1):
        f1.write("\t" + str(xbarra[j][k][0]) + "," + str(xbarra[j][k][1]) + "," + str(xbarra[j][k][2]) + "," + str(xbarra[j][k][3]))
             # "  " + str(x_barra[k + 1][0]) + "," + str(x_barra[k + 1][1]) + "," + str(x_barra[k + 1][2]) + "," + str(x_barra[k + 1][3]) + "    " + str(x_barra[k + 2][0]) + "," + str(x_barra[k + 2][1]) + "," + str(x_barra[k + 2][2]) + "," + str(x_barra[k + 2][3]) + "\n")
    f1.write("\n")



print("terminou")
