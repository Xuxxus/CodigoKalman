import numpy as np
from numpy import dot
import math
from filterpy.kalman import predict, update
import matplotlib.pyplot as plt


pip install filterpy

float_formartter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formartter})
angleyawsg = [] #altereiiiiiiiii
anglepitchs = []
anglepitchsg = [] #altereiiiiiiiii
anglerolls = []
anglerollsg = [] #altereiiiiiiiii
raterollsg =[]   #altereiiiiiiiii
ratepitchsg = []   #altereiiiiiiiii
rateyawsg = []   #altereiiiiiiiii
x_barra1 = []
x_barrasg =[]
kalman0g= [] #altereiii
kalman0grad= [] #altereiii
kalman0 = []
kalman1g= [] #altereiiii
kalman1grad= [] #altereiii
kalman1 = []
kalman2g = [] # altereiiiiiiiii
kalman2grad= [] #altereiii
tempos = []
temposg = [] #altereiiiiiiiii
tempografico = []
quartg=[] #altereiii
q1=[]
q2=[]
tempozero = 0 #altereiii
posinicialg = []  #altereiiiii
iniciolista =[]
acelX = []
acelY =[]
acelZ =[]
gyroX =[]
gyroY =[]
gyroZ =[]
kX =[]
kY = []
kZ = []
g = []
quart =[]

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
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)





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



i = 0
count = 0
x_barra = []

tempo = []
'''
rateCalibrationRoll = [-0.13, -2.85, 1.69, -4.2, 7.55, -3.95, 2.65, -26.7, -1.0, -3.65, 0.0, 0.0, 0.0]
rateCalibrationPitch = [0.83, 1.68, -0.89, 2.45, 1.07, -2.74, -3.26, -1.29, 0.72, -0.31, 0.0, 0.0, 0.0]
rateCalibrationYaw = [0.6, 0.53, 0.24, -1.15, 1.87, -0.34, -0.59, 0.65, 0.62, 0.45, 0.0, 0.0, 0.0]'''

rateCalibrationRoll = [-0.09, -3.68, -4.72, -4.2, 7.55, -3.95, 2.65, -26.7, -1.0, -3.65, 0.0, 0.0, 0.0]
rateCalibrationPitch = [0.63, -2.89, 1.93, 2.45, 1.07, -2.74, -3.26, -1.29, 0.72, -0.31, 0.0, 0.0, 0.0]
rateCalibrationYaw = [0.44, -0.42, -0.91, -1.15, 1.87, -0.34, -0.59, 0.65, 0.62, 0.45, 0.0, 0.0, 0.0]



#24.11
#acelCalibrationRoll = [1.04, 1.03, 1.03, 1.03, 1.0, 1.03, 1.06, 1.04, 1.03, 0.96, 0.0, 0.0, 0.0]
#acelCalibrationPitch = [0.98, 1.0, 1.0, 1.0, 0.99, 1.0, 0.99, 0.98, 0.99, 1.01, 0.0, 0.0, 0.0]
#acelCalibrationYaw = [0.95, 0.99, 1.0, 0.95, 0.93, 0.99, 0.74, 1.0, 0.95, 0.89, 0.0, 0.0, 0.0]

acelCalibrationRoll = [1.030966, 1.027678, 1.041764,  1.074233, 1.042983, 1.03, 1.06, 1.04, 1.03, 0.96, 0.0, 0.0, 0.0]
acelCalibrationPitch =[1.003606, 0.983924, 0.987624, 0.98639, 0.98764, 1.0, 0.99, 0.98, 0.99, 1.01, 0.0, 0.0, 0.0]
acelCalibrationYaw = [0.96429, 1.072670, 0.972605,  0.749889, 1.014723, 0.99, 0.74, 1.0, 0.95, 0.89, 0.0, 0.0, 0.0]



# dados = [1,2,3,4,5,6,7,8,9,2,1,2,3,4]
# Impelmentar busca de arquivos e leitura (pandas)
# Criar for com loop para pegar a cada 7 dados fornecidos pelo cartão SD (Traduzir do código C#)
# Multiplicação de matrizes feita, agora ver transformação para quaternion e depois manipulação das strings

c = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])



x = np.array([[0.0],
              [0.0],
              [0.0],
              [0.0],
              [0.0],
              [0.0]])

u = np.array([[0.0],     #altereiii
              [0.0],
              [0.0]])

a_shape = (6, 6)
P = np.zeros(a_shape)

nstate_nobs = (6, 3)
nobs_nobs = (3, 3)
n_z = (3,1)

S = np.zeros(nobs_nobs)
k = np.zeros(nstate_nobs)
z = np.zeros(n_z)

with open('take2.txt') as f:
    contents = f.read()
    dados = contents.split(',')
    print(dados)
    print(len(dados)) #altereiiiiiiiii
    print(len(dados)/7) #altereiiiiiiiii

n_sensor = int(input("Insira o número de sensores: "))
nome_sensores = [] #altereiiiiiiiii
linhasg = ((len(dados))//7) #altereiiiiiiiii
linhasporsensor = linhasg//n_sensor
dadoscompletos = linhasporsensor*7*n_sensor # controlo a quantidade de dados para garantir o mesmo intervalo de amostras que possuo de cada IMU

for i in range(n_sensor):
    nome_sensores.append(str(input("Nome do sensor %d: " %(i+1))))




for num in range(0,n_sensor,1): # roda um for para cada sensor existeste. Define a posição do primeiro dado na lista de dados
  #zerando as variaveis pois estou rodando os calculos e o Kalman 1 IMU por vez completo
  kalman0g = []
  kalman1g = []
  kalman2g = []
  kalman0grad =[]
  kalman1grad = []
  kalman2grad =[]
  quartg = []
  anglerollsg = []
  anglepitchsg =[]
  angleyawsg =[]
  raterollsg =[]
  ratepitchsg=[]
  rateyawsg=[]
  ciclo = 0

  x = np.array([[0.0],    #zerando os parametros de Kalman
              [0.0],
              [0.0],
              [0.0],
              [0.0],
              [0.0]])
  u = np.array([[0.0],     #altereiii
              [0.0],
              [0.0]])

  #Ti = 0.2
  contador =6



  P = np.zeros(a_shape)
  S= np.zeros(nobs_nobs)
  k = np.zeros(nstate_nobs)
  z = np.zeros(n_z)

  if (num ==0):#sensor 1
    print("Sensor 1")
    inicio =0
    count = 0

        # parametros obtidos experimentalmente analisando os gráficos plotados
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1#expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido



  if (num==1):#sensor 2
    print("Sensor 2")
    inicio =7
    count = count + 1 # valor referente à posição do vetor dos dados de calibração do sensor 2

    # parametros obtidos experimentalmente analisando os gráficos plotados
    #acel
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    #gyro
    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1 #expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido


  if (num==2):#sensor 3
    inicio =14
    print("Sensor 3")
    count = count+1

        # parametros obtidos experimentalmente analisando os gráficos plotados
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1 #expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido


  if (num==3):#sensor 4
    inicio =21
    print("Sensor 4")
    count = count+1

        # parametros obtidos experimentalmente analisando os gráficos plotados
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1 #expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido

  if (num==4):#sensor 5
    inicio =28
    print("Sensor 5")
    count = count+1

        # parametros obtidos experimentalmente analisando os gráficos plotados
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1 #expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido


  if (num==5):#sensor 6
    inicio =35
    print("Sensor 6")
    count = count+1

        # parametros obtidos experimentalmente analisando os gráficos plotados
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1 #expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido


  if (num==6):#sensor 7
    inicio =42
    print("Sensor 7")
    count = count+1

        # parametros obtidos experimentalmente analisando os gráficos plotados
    n_roll_real = 3
    n_pitch_real = 3
    n_yaw_real = 3 #3 # quanto menor, mais estavel ele fica, tende a permanecer no mesmo lugar

    m_roll_real = 5  #quanto menor, mais eu confio no giroscópio
    m_pitch_real = 5
    m_yaw_real = 0.1 #expirmentalmente:0.001 #0.1 # quanto menor, menos eu confio nesse sinal , gera-se mais ruido





  #iniciolista.append(ciclo)  # lista que armazena a posição dos dados dos sensores
  r = np.array([[n_roll_real, 0.0, 0.0],
              [0.0, n_pitch_real, 0.0],
              [0.0, 0.0, n_yaw_real]])


  for i in range(inicio+(7*n_sensor), (dadoscompletos-(7*n_sensor)), 7*n_sensor): # roda o for a partir da 2º amostrar até a quantidade de dados completos menos a ultima linha de cada sensor

      Ti =  (float(dados[i+6]) - float(dados[i+6-(7*n_sensor)])) # Ti é o delta T entre a coleta atual e a coleta anterior

      q = np.array([[m_roll_real * Ti * Ti, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, m_pitch_real *Ti * Ti, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, m_yaw_real *Ti *Ti, 0.0, 0.0, 0.0],
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

      if num ==0:
        angleRoll = -10+(math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        anglePitch = -105+ (-math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
              ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                      (float(dados[i + 2]) ) * (
                      float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        angleYaw =  -80+(math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))*(1/(3.14159265/180))

      if num ==1:
        angleRoll = -10+ (math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        anglePitch = -100+ (-math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
              ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                      (float(dados[i + 2]) ) * (
                      float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        angleYaw = -100+(math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))*(1/(3.14159265/180))

      if num ==2:
        angleRoll = 0+(math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))*(1/(3.14159265/180)))



        anglePitch = -110+ (-math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
              ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                      (float(dados[i + 2]) ) * (
                      float(dados[i + 2]) ))))*(1/(3.14159265/180)))


        angleYaw =  -70+(math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))*(1/(3.14159265/180))

      if num ==3:
        angleRoll = -20+(math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        anglePitch = -110+ (-math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
              ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                      (float(dados[i + 2]) ) * (
                      float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        angleYaw =  -90+(math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))*(1/(3.14159265/180))


      if num ==4:
        angleRoll = -90+(math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        anglePitch = 90+ (-math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
              ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                      (float(dados[i + 2]) ) * (
                      float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        angleYaw =  0+(math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))*(1/(3.14159265/180))

      if num==5:
        angleRoll = 0+(math.atan((float(dados[i + 1]) ) / math.sqrt(
            (float(dados[i])) * (float(dados[i]) ) + (
                    (float(dados[i + 2]) ) * (
                    float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        anglePitch = 0+ (-math.atan((float(dados[i]) ) / math.sqrt(      #altereiiiiiiiii
              ((float(dados[i + 1]) ) * (float(dados[i + 1]) )) + (
                      (float(dados[i + 2]) ) * (
                      float(dados[i + 2]) ))))*(1/(3.14159265/180)))

        angleYaw =  0+(math.atan(((float(dados[i + 2]))/math.sqrt((float(dados[i]))*(float(dados[i]))+(float(dados[i+1]))*(float(dados[i+1]))))))*(1/(3.14159265/180))






      anglerollsg.append(angleRoll)   #altereiiiiiiiii
      anglepitchsg.append(anglePitch)   #altereiiiiiiiii

      #temposg.append(i/700) # i vai de 7 em 7   #altereiiiiiiiii



      TT =  (float(dados[i+6]) - float(dados[6])) # TT é o tempo desde o começo do programa

      if num ==0:
        tempografico.append((float(dados[i+6]) - float(dados[6])))
      if num == (n_sensor-1):      #o tempo do arquivo sto será TT
        temposg.append(float(TT))

      #contador = contador +7
      #f1.write(str(float(dados[suporte+7]) - float(dados[6])))

      #angleYaw =




      z[0] = angleRoll
      z[1] = anglePitch
      z[2] = angleYaw

      angleyawsg.append(angleYaw)   #altereiiiiiiiii

      # rateRoll = (Convert.ToDouble(list[i + 3]) / 131) - rateCalibrationRoll[count];
      # ratePitch = (Convert.ToDouble(list[i + 4]) / 131) - rateCalibrationPitch[count];
      # rateYaw = (Convert.ToDouble(list[i + 5]) / 131) - rateCalibrationYaw[count];

      rateRoll = ((float(dados[i + 3]) ) - rateCalibrationRoll[count])   #altereiiiiiiiii
      ratePitch =  ((float(dados[i + 4]) ) - rateCalibrationPitch[count])  #altereiiiiiiiii
      rateYaw = ((float(dados[i + 5]) ) - rateCalibrationYaw[count])  #altereiiiiiiiii

      raterollsg.append(rateRoll)
      ratepitchsg.append(ratePitch)
      rateyawsg.append(rateYaw)
      #print("Giroscopio")
      #print(rateRoll)
      #print(ratePitch)
      #print(rateYaw)

      u[0] = rateRoll
      u[1] = ratePitch
      u[2] = rateYaw



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

      is_nosingular = determinantOfMatrix(S, 2)

      k = dot(dot(P, c.T), S)

      if is_nosingular:
          #print("Entrou no Kalman")
          x, P = update(x, P, z, r, c)
          #print("----------------Update-----------------")
          #print('x =', x)
          #print('P =', P)
          #print("update resultado")
          #print(x)
          #print("kalman0g ")

          kalman0g.append((x[0][0])) #altereiiiiiiiiii

          #print(kalman0g)
          #print("xbarra 1 ")
          kalman1g.append((x[1][0])) #altereiiiiiiiiiii
          #print(kalman1g)
          kalman2g.append( (x[2][0]))
          kalman0grad.append( (float(kalman0g[ciclo])*(3.14159265/180)))  #altereiiiiiiiiiii
          kalman1grad.append((float(kalman1g[ciclo])*(3.14159265/180)))  #altereiiiiiiiiiii
          kalman2grad.append((float(kalman2g[ciclo])*(3.14159265/180)))  #altereiiiiiiiiiii
          quartg.append(get_quaternion_from_euler(float(kalman0grad[ciclo]),float(kalman1grad[ciclo]),float(kalman2grad[ciclo])))

          #print(quartg[0][1])
          ciclo = ciclo +1





  #tratando os dados para plotar nos graficos
  acelX.append(anglerollsg)
  acelY.append(anglepitchsg)
  acelZ.append(angleyawsg)
  gyroX.append(raterollsg)
  gyroY.append(ratepitchsg)
  gyroZ.append(rateyawsg)
  kX.append(kalman0g)
  kX.append(kalman0grad)
  kY.append(kalman1g)
  kY.append(kalman1grad)
  kZ.append(kalman2g)
  kZ.append(kalman2grad)
  quart.append(quartg)
  print("2 tabs")

q1.append(get_quaternion_from_euler(float(0.0),float(-1.48352),float(-1.48352)))
print(q1)

q1.append(get_quaternion_from_euler(float(-1.48352),float(0.0),float(1.48352)))
print(q1)
f1 = open('posicao_inicial.sto', 'w')   # estrutura: 1 tempo e os quartenions de todos os sensores
f1.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4\nendheader\n")
f1.write('time')
for l in range(len(nome_sensores)):
    f1.write('\t' + nome_sensores[l])
f1.write("\n")
f1.write(str(temposg[50])) #tempo da 5º amostra do ultimo sensor(tempo necessário para o Kalman estabilizar)
for ll in range(0,len(nome_sensores),1):
  #f1.write("\t" + str(quartg[(iniciolista[ll])+15][0]) + "," + str(quartg[(iniciolista[ll])+15][1]) + "," + str(quartg[(iniciolista[ll])+15][2]) + "," + str(quartg[(iniciolista[ll])+15][3]))
  f1.write("\t" + str(quart[ll][50][0]) + "," + str(quart[ll][50][1]) + "," + str(quart[ll][50][2]) + "," + str(quart[ll][50][3]))

  #f1.write("\t" + str(q1[ll][0]) + "," + str(q1[ll][1]) + "," + str(q1[ll][2]) + "," + str(q1[ll][3]))

f1 = open('opensimTeste.sto', 'w')   # estrutura: 1 tempo -> quartenions de todos os sensores     tempo -> quartenions de todos os sensores
f1.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4\nendheader\n")
f1.write('time')
for l in range(len(nome_sensores)):
    f1.write('\t' + nome_sensores[l])
f1.write("\n")
suporte = 6   #nao estou considerando os primeiros 2s (154 dados) e nem a ultima linha de dados
for k in range(20, len(temposg)-1,1): #vai de 20 até o numero de amostras coletadas (temposg) de 1 em 1
  f1.write(str(temposg[k]))
  for ll in range(0,len(nome_sensores),1):
    suporte = suporte +7
    #f1.write("\t" + str(quartg[(iniciolista[ll])+k][0]) + "," + str(quartg[(iniciolista[ll])+k][1]) + "," + str(quartg[(iniciolista[ll])+k][2]) + "," + str(quartg[(iniciolista[ll])+k][3]))
    f1.write("\t" + str(quart[ll][k][0]) + "," + str(quart[ll][k][1]) + "," + str(quart[ll][k][2]) + "," + str(quart[ll][k][3]))
  f1.write("\n")



print("terminou")
 # Plot gráfico Angulo acelerometro em graus Sensor n
# @title Plot gráfico Angulo acelerometro em graus Sensor n
n = 1  #n=0 -> angulo acelerometro IMU 0 em graus,   n=1 -> angulo acelerometro  IMU 1 em graus ...
b = 1
plt.figure().set_figwidth(20)
plt.plot(tempografico,acelX[n], color ='y') # amarelo
plt.plot(tempografico,acelY[n], color ='b') # azul escuro
plt.plot(tempografico,acelZ[n],color = 'm') #rosa
plt.plot(tempografico,acelX[b], color ='y') # amarelo
plt.plot(tempografico,acelY[b], color ='b') # azul escuro
plt.plot(tempografico,acelZ[b],color = 'm') #verde
plt.xlabel("Tempo [s]")
plt.ylabel("Graus [º]")
plt.xticks(np.arange(0, 40, step=1))
plt.yticks(np.arange(-90, 90, step=10))
plt.grid(color ='g', linestyle = '--', linewidth = 0.5)
#tempografico

# Plot gráfico ângulo acelerometro x angulo real kalman em graus
# @title Plot gráfico angulo acelerometro x Angulo real Kalman em graus
a = 0
n = 0 #n=0 -> kalman IMU 0 em graus,   n=1 -> kalman IMU 0 em rad   n=2 -> kalman IMU 1 em graus ...
#a 0 1 2 3 4
#n 0 2 4 6 8
plt.figure().set_figwidth(20)
plt.plot(tempografico,acelX[a], color ='r',label = 'AcelX') #vermelho
plt.plot(tempografico,acelY[a], color ='b',label = 'AcelY') # azul escuro
plt.plot(tempografico,acelZ[a], color ='g',label ='AcelZ') #verde
plt.plot(tempografico,kX[n], color ='y',label ='KalmanX') # amarelo
plt.plot(tempografico,kY[n],color = 'c',label ='KalmanY' ) # azul claro
plt.plot(tempografico,kZ[n],color = 'm',label ='KalmanZ') #rosa
plt.xlabel("Tempo [s]")
plt.ylabel("Graus [º]")
plt.legend()
plt.xticks(np.arange(0, 24, step=1))
plt.yticks(np.arange(-240, 90, step=10))
plt.grid(color ='g', linestyle = '--', linewidth = 0.5)

# Plot gráfico ângulo acelerometro x angulo real kalman em graus
# @title Plot gráfico angulo acelerometro x Angulo real Kalman em graus
#a = 1
#n = 2 #n=0 -> kalman IMU 0 em graus,   n=1 -> kalman IMU 0 em rad   n=2 -> kalman IMU 1 em graus ...
plt.figure().set_figwidth(20)
plt.plot(tempografico,acelY[0], color ='r') #vermelho
plt.plot(tempografico,acelY[1], color ='r') #vermelho
#plt.plot(tempografico,acelY[a], color ='b') # azul escuro
#plt.plot(tempografico,acelZ[a], color ='g') #verde
plt.plot(tempografico,kY[0], color ='y') # amarelo
plt.plot(tempografico,kY[2], color ='y') # amarelo
#plt.plot(tempografico,kY[n],color = 'c' ) # azul claro
#plt.plot(tempografico,kZ[n],color = 'm') #rosa
plt.xlabel("Tempo [s]")
plt.ylabel("Graus [º]")
plt.xticks(np.arange(0, 40, step=1))
plt.yticks(np.arange(-80, 90, step=10))
plt.grid(color ='g', linestyle = '--', linewidth = 0.5)
plt.axis([0,40,-90,30]) #escolhe o intervalo/recorte que ira mostrar o grafico

#Plot gráfico angulo kalman em graus
# @title Plot gráfico angulo  Kalman em graus

n = 6 #n=0 -> kalman IMU 0 em graus,   n=1 -> kalman IMU 0 em rad   n=2 -> kalman IMU 1 em graus ...
plt.figure().set_figwidth(20)
m=6
plt.plot(tempografico,kX[n], color ='y') # amarelo
plt.plot(tempografico,kY[n],color = 'c' ) # azul claro
plt.plot(tempografico,kZ[n],color = 'm') #rosa
plt.plot(tempografico,kX[m], color ='y') # amarelo
plt.plot(tempografico,kY[m],color = 'c' ) # azul claro
plt.plot(tempografico,kZ[m],color = 'm') #rosa
plt.xlabel("Tempo [s]")
plt.ylabel("Graus [º]")
plt.xticks(np.arange(0, 70, step=1))
plt.yticks(np.arange(-270, 100, step=10))
plt.grid(color ='g', linestyle = '--', linewidth = 0.5)
plt.axis([6,45,-270,120]) #escolhe o intervalo/recorte que ira mostrar o grafico

pip install filterypy

#Plot gráfico giroscopio

# @title Plot gráfico giroscopio
n=1
plt.figure().set_figwidth(20)
plt.plot(tempografico,gyroX[n], color ='r') #vermelho
plt.plot(tempografico,gyroY[n], color ='b') # azul escuro
plt.plot(tempografico,gyroZ[n], color ='y') #amarelo

plt.xlabel("Tempo [s]")
plt.ylabel("Graus [º]")
plt.xticks(np.arange(0, 60, step=1))
plt.yticks(np.arange(-100, 100, step=20))
plt.grid(color ='g', linestyle = '--', linewidth = 0.5)
#plt.axis([0,15,-20,70]) #escolhe o intervalo/recorte que ira mostrar o grafico