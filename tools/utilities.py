# utilities.py

def read_matrix(filename):
   infile = open(filename,'r')
   return [list(map(float,row.split())) for row in infile.readlines()]

def tidy_up(matrix):
   out = []
   for row in matrix:
      tmp = []
      for value in row:
         if round(value) == round(value,8):
            tmp.append(round(value))
         else:
            tmp.append(round(value,3))
      out.append(tmp)
   return out

def scalar_multiply(matrix,scalar):
   for i in range(len(matrix)):
      for j in range(len(matrix[i])):
         matrix[i][j] *= scalar
   return matrix